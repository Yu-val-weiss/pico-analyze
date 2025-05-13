"""
Simple components are those that are a single layer. For example, the weight matrix of a layer is
a single component. In other words, simple components are those that can just be extracted directly
from the stored out checkpoint data without much additional computation.
"""

from functools import lru_cache
from typing import Any, Dict, Tuple

import torch

from src.components._registry import register_component
from src.components.base_lora import BaseLoraComponent
from src.config.base import BaseComponentConfig
from src.utils.exceptions import InvalidComponentError


@register_component("lora_ov_circuit")
class LoraOVComponent(BaseLoraComponent):
    """
    LoraComponent multiplies the lora parameters at each checkpoint.
    """

    def __init__(self, training_config: Dict[str, Any]):
        super().__init__(training_config)

        self.d_model = training_config["model"]["d_model"]

        self.attention_n_heads = training_config["model"]["attention_n_heads"]
        self.attention_n_kv_heads = training_config["model"]["attention_n_kv_heads"]
        self.attention_head_dim = self.d_model // self.attention_n_heads

    def validate_component(self, component_config: BaseComponentConfig) -> None:
        """
        Lora components can only be weights.
        """
        if component_config.data_type not in ("weights", "activations"):
            raise InvalidComponentError(
                f"Simple component only supports weights not {component_config.data_type}."
            )
        if (
            "value_layer" not in component_config.layer_suffixes
            or "output_layer" not in component_config.layer_suffixes
        ):
            raise InvalidComponentError(
                "Lora OV circuit component requires value and output layer suffixes."
            )

    @lru_cache(maxsize=50)
    def compute_ov_weights(
        self, layer_v_proj: torch.Tensor, layer_o_proj: torch.Tensor
    ):
        """Compute ov weights for a lora module.
        Expects the pre-computed matrix for each projection."""

        layer_ov_weights_ph = {}  # layer weights per head

        for head_idx in range(self.attention_n_heads):
            kv_head_idx = head_idx // (
                self.attention_n_heads // self.attention_n_kv_heads
            )

            start_v_proj = kv_head_idx * self.attention_head_dim
            end_v_proj = (kv_head_idx + 1) * self.attention_head_dim
            vp_sl = slice(start_v_proj, end_v_proj)

            start_o_proj = head_idx * self.attention_head_dim
            end_o_proj = (head_idx + 1) * self.attention_head_dim
            op_sl = slice(start_o_proj, end_o_proj)

            ov_weights_per_head = layer_v_proj[vp_sl, :] @ layer_o_proj[:, op_sl]

            layer_ov_weights_ph[f"{head_idx}"] = ov_weights_per_head

        layer_ov_weights = torch.cat(list(layer_ov_weights_ph.values()), dim=1)

        return layer_ov_weights_ph, layer_ov_weights

    @lru_cache(maxsize=50)
    def compute_ov_activations(
        self,
        layer_value_activation: torch.Tensor,
        layer_output_projection: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute the OV activations for a single layer. Uses a cache to speed up the computation,
        if the component is used across multiple metrics.

        NOTE: the OV-Circuit operates 'per head' of the attention module, so we compute the OV
        activations for each head separately and then concatenate them together.

        Args:
            layer_value_activation: The value activations for the layer.
            layer_output_projection: The output projection for the layer.

        Returns:
            A tuple:
                - A dictionary mapping head indices to OV component activations.
                - A concatenated tensor of the OV component activations.
        """
        layer_ov_activation_per_head = {}

        for head_idx in range(self.attention_n_heads):
            kv_head_idx = head_idx // (
                self.attention_n_heads // self.attention_n_kv_heads
            )

            if layer_value_activation.dtype != layer_output_projection.dtype:
                # NOTE: activations might be stored as memory efficient floats (e.g. bfloat16)
                # so we need to make sure we cast to the same type as the weights
                layer_value_activation = layer_value_activation.to(
                    layer_output_projection.dtype
                )

            start_value_activation = kv_head_idx * self.attention_head_dim
            end_value_activation = (kv_head_idx + 1) * self.attention_head_dim

            ov_activation_per_head = (
                layer_value_activation[:, start_value_activation:end_value_activation]
                @ layer_output_projection[
                    :,
                    head_idx * self.attention_head_dim : (head_idx + 1)
                    * self.attention_head_dim,
                ].T
            )

            layer_ov_activation_per_head[f"{head_idx}"] = ov_activation_per_head

        layer_ov_activation = torch.cat(
            list(layer_ov_activation_per_head.values()), dim=1
        )

        return layer_ov_activation_per_head, layer_ov_activation

    def __call__(
        self,
        checkpoint_states: Dict[str, Dict[str, torch.Tensor]],
        component_config: BaseComponentConfig,
    ) -> Dict[str, Any]:
        """
        Given a dictionary of checkpoint data, extract the weights for the given layer suffix and layer.

        Args:
            checkpoint_states: Checkpoint data (activations, weights, gradients)
            component_config: The component configuration.

        Returns:
            A dictionary mapping layer names to MLP activations.
        """
        layer_suffixes = component_config.layer_suffixes
        output_layer_id = layer_suffixes["output_layer"]
        value_layer_id = layer_suffixes["value_layer"]

        checkpoint_layer_component = {}

        _data = checkpoint_states[component_config.data_type]
        _weights = checkpoint_states["weights"]
        _model_prefix = self.get_model_prefix(checkpoint_states)

        for layer_idx in component_config.layers:
            value_layer_prefix = f"{_model_prefix}{layer_idx}.{value_layer_id}"
            output_layer_prefix = f"{_model_prefix}{layer_idx}.{output_layer_id}"

            # NOTE: each computation is done 'per head' of the attention module, and return a
            # tuple of a dictionary mapping head indices to the OV component and a concatenated
            # tensor of the OV component

            if component_config.data_type == "weights":
                # here it should be projection
                lora_layer_v_val = (
                    _data[f"{value_layer_prefix}.B_lora"]
                    @ _data[f"{value_layer_prefix}.A_lora"]
                )
            else:
                # for activation it should be the activation itself
                # this is simply the activation at B_lora
                lora_layer_v_val = _data[f"{value_layer_prefix}.B_lora"]

            lora_layer_o_proj = (
                _weights[f"{output_layer_prefix}.B_lora"]
                @ _weights[f"{output_layer_prefix}.A_lora"]
            )

            # LORA
            if component_config.data_type == "weights":
                lora_ov_comp_ph, lora_ov_comp = self.compute_ov_weights(
                    lora_layer_v_val, lora_layer_o_proj
                )
            else:
                lora_ov_comp_ph, lora_ov_comp = self.compute_ov_activations(
                    lora_layer_v_val, lora_layer_o_proj
                )

            for head_idx, ov_component_head in lora_ov_comp_ph.items():
                checkpoint_layer_component[
                    f"{_model_prefix}{layer_idx}.ov_circuit.lora.{component_config.data_type}.heads.{head_idx}"
                ] = ov_component_head

            checkpoint_layer_component[
                f"{_model_prefix}{layer_idx}.ov_circuit.lora.{component_config.data_type}"
            ] = lora_ov_comp

            # BASE

            base_layer_v_val = _data[value_layer_prefix]
            base_layer_o_proj = _weights[output_layer_prefix]

            if component_config.data_type == "weights":
                base_ov_comp_ph, base_ov_comp = self.compute_ov_weights(
                    base_layer_v_val, base_layer_o_proj
                )
            else:
                base_ov_comp_ph, base_ov_comp = self.compute_ov_activations(
                    base_layer_v_val, base_layer_o_proj
                )

            for head_idx, ov_component_head in base_ov_comp_ph.items():
                checkpoint_layer_component[
                    f"{_model_prefix}{layer_idx}.ov_circuit.base.{component_config.data_type}.heads.{head_idx}"
                ] = ov_component_head

            checkpoint_layer_component[
                f"{_model_prefix}{layer_idx}.ov_circuit.base.{component_config.data_type}"
            ] = base_ov_comp

            # FULL
            full_layer_v_val = lora_layer_v_val * self.lora_s + base_layer_v_val
            full_layer_o_proj = lora_layer_o_proj * self.lora_s + base_layer_o_proj

            if component_config.data_type == "weights":
                full_ov_comp_ph, full_ov_comp = self.compute_ov_weights(
                    full_layer_v_val, full_layer_o_proj
                )
            else:
                full_ov_comp_ph, full_ov_comp = self.compute_ov_activations(
                    full_layer_v_val, full_layer_o_proj
                )

            for head_idx, ov_component_head in full_ov_comp_ph.items():
                checkpoint_layer_component[
                    f"{_model_prefix}{layer_idx}.ov_circuit.full.{component_config.data_type}.heads.{head_idx}"
                ] = ov_component_head

            checkpoint_layer_component[
                f"{_model_prefix}{layer_idx}.ov_circuit.full.{component_config.data_type}"
            ] = full_ov_comp

        return checkpoint_layer_component
