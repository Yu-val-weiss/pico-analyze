"""
Output-Value Circuit component: a compound component that is composed of the value and output
projections of the attention modules of a transformer model.
"""

from functools import lru_cache
from typing import Any, Dict, Tuple

import torch

from src.components._registry import register_component
from src.components.base import BaseComponent
from src.config.base import BaseComponentConfig
from src.utils.exceptions import InvalidComponentError


@register_component("ov_circuit")
class OVComponent(BaseComponent):
    """
    Compute the 'OV-Circuit' component. The idea of an OV-Circuit stems from the observation
    that in an attention module, the value and the output projections always operate jointly,
    and write into the 'residual stream'. Thus, it makes sense to treat the Output and Value
    matrices as a single 'OV-Circuit' matrix, which we can compute the activations and weights for.

    Also note that the OV-Circuit operates 'per head' of the attention module.

    To read more about the OV-Circuit, see:
        https://transformer-circuits.pub/2021/framework/index.html
    """

    def __init__(self, training_config: Dict[str, Any]):
        super().__init__(training_config)

        self.d_model = training_config["model"]["d_model"]

        self.attention_n_heads = training_config["model"]["attention_n_heads"]
        self.attention_n_kv_heads = training_config["model"]["attention_n_kv_heads"]
        self.attention_head_dim = self.d_model // self.attention_n_heads

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

    @lru_cache(maxsize=50)
    def compute_ov_weights(
        self,
        layer_value_projection: torch.Tensor,
        layer_output_projection: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute the OV weights for a single layer. Uses a cache to speed up the computation,
        if the component is used across multiple metrics.

        NOTE: the OV-Circuit operates 'per head' of the attention module, so we compute the OV
        weights for each head separately and then concatenate them together.

        Args:
            layer_value_projection: The value projection for the layer.
            layer_output_projection: The output projection for the layer.

        Returns:
            A tuple:
                - A dictionary mapping head indices to OV component weights.
                - A concatenated tensor of the OV component weights.
        """

        layer_ov_weights_per_head = {}

        for head_idx in range(self.attention_n_heads):
            kv_head_idx = head_idx // (
                self.attention_n_heads // self.attention_n_kv_heads
            )

            start_value_projection = kv_head_idx * self.attention_head_dim
            end_value_projection = (kv_head_idx + 1) * self.attention_head_dim

            start_output_projection = head_idx * self.attention_head_dim
            end_output_projection = (head_idx + 1) * self.attention_head_dim

            ov_weights_per_head = (
                layer_value_projection[start_value_projection:end_value_projection, :]
                @ layer_output_projection[
                    :, start_output_projection:end_output_projection
                ]
            )

            layer_ov_weights_per_head[f"{head_idx}"] = ov_weights_per_head

        layer_ov_weights = torch.cat(list(layer_ov_weights_per_head.values()), dim=1)

        return layer_ov_weights_per_head, layer_ov_weights

    @lru_cache(maxsize=50)
    def compute_ov_gradients(
        self,
        layer_value_gradient: torch.Tensor,
        layer_output_gradient: torch.Tensor,
        layer_value_projection: torch.Tensor,
        layer_output_projection: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute the OV gradients for a single layer. Uses a cache to speed up the computation,
        if the component is used across multiple metrics.

        NOTE: the OV-Circuit operates 'per head' of the attention module, so we compute the OV
        gradients for each head separately and then concatenate them together.

        The gradients are computed using the chain rule and is composed of two parts:
            1. dL/dOutput * dOutput/dOV
            2. dL/dValue * dValue/dOV

        Args:
            layer_value_gradient: The gradient of the value projection for the layer.
            layer_output_gradient: The gradient of the output projection for the layer.
            layer_value_projection: The value projection for the layer.
            layer_output_projection: The output projection for the layer.

        Returns:
            A tuple:
                - A dictionary mapping head indices to OV component gradients.
                - A concatenated tensor of the OV component gradients.
        """

        target_dtype = layer_output_projection.dtype
        layer_value_gradient = layer_value_gradient.to(target_dtype)
        layer_output_gradient = layer_output_gradient.to(target_dtype)
        layer_value_projection = layer_value_projection.to(target_dtype)

        layer_ov_gradients_per_head = {}

        for head_idx in range(self.attention_n_heads):
            kv_head_idx = head_idx // (
                self.attention_n_heads // self.attention_n_kv_heads
            )

            start_value_projection = kv_head_idx * self.attention_head_dim
            end_value_projection = (kv_head_idx + 1) * self.attention_head_dim

            start_output_projection = head_idx * self.attention_head_dim
            end_output_projection = (head_idx + 1) * self.attention_head_dim

            head_value_gradient = layer_value_gradient[
                start_value_projection:end_value_projection, :
            ]

            head_output_gradient = layer_output_gradient[
                :, start_output_projection:end_output_projection
            ]

            head_value_projection = layer_value_projection[
                start_value_projection:end_value_projection, :
            ]

            head_output_projection = layer_output_projection[
                :, start_output_projection:end_output_projection
            ]

            # NOTE: chain rule for gradients; dL/dOV = dL/dOutput * dOutput/dOV + dL/dValue * dValue/dOV
            head_ov_gradient = (head_output_projection @ head_value_gradient) + (
                head_output_gradient @ head_value_projection
            )

            layer_ov_gradients_per_head[f"{head_idx}"] = head_ov_gradient

        layer_ov_gradients = torch.cat(
            list(layer_ov_gradients_per_head.values()), dim=1
        )

        return layer_ov_gradients_per_head, layer_ov_gradients

    def validate_component(self, component_config: BaseComponentConfig) -> None:
        """
        OV circuit components require value and output layer suffixes to be specified.
        """
        if (
            "value_layer" not in component_config.layer_suffixes
            or "output_layer" not in component_config.layer_suffixes
        ):
            raise InvalidComponentError(
                "OV circuit component requires value and output layer suffixes."
            )

    def __call__(
        self,
        checkpoint_states: Dict[str, Dict[str, torch.Tensor]],
        component_config: BaseComponentConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates the OV circuit component. The OV circuit component is a compound component
        that is composed of the value and output projections of the model.

        NOTE: the OV-Circuit operates 'per head' of the attention module, so we compute and return
          the OV activations, weights, and gradients for each head separately and then concatenate
          them together as well into one aggregated tensor.

        Args:
            checkpoint_states: Dict[str, Dict[str, torch.Tensor]] -- the checkpoint states
                to generate the component for.
            component_config: BaseComponentConfig -- the component configuration.

        Returns:
            Dict[str, torch.Tensor] -- the OV circuit component; mapping layer names to OV circuit
                activations.
        """

        layer_suffixes = component_config.layer_suffixes

        checkpoint_layer_component = {}

        _model_prefix = self.get_model_prefix(checkpoint_states)

        for layer_idx in component_config.layers:
            layer_output_projection = checkpoint_states["weights"][
                f"{_model_prefix}{layer_idx}.{layer_suffixes['output_layer']}"
            ]

            # NOTE: depending on whether we are computing activations, weights, or gradients, we
            # need to fetch the appropriate tensors from the checkpoint states

            if component_config.data_type == "activations":
                layer_value_activation = checkpoint_states["activations"][
                    f"{_model_prefix}{layer_idx}.{layer_suffixes['value_layer']}"
                ]
            elif component_config.data_type in ["weights", "gradients"]:
                layer_value_projection = checkpoint_states["weights"][
                    f"{_model_prefix}{layer_idx}.{layer_suffixes['value_layer']}"
                ]

            if component_config.data_type == "gradients":
                layer_output_gradient = checkpoint_states["gradients"][
                    f"{_model_prefix}{layer_idx}.{layer_suffixes['output_layer']}"
                ]
                layer_value_gradient = checkpoint_states["gradients"][
                    f"{_model_prefix}{layer_idx}.{layer_suffixes['value_layer']}"
                ]

            # NOTE: each computation is done 'per head' of the attention module, and return a
            # tuple of a dictionary mapping head indices to the OV component and a concatenated
            # tensor of the OV component

            if component_config.data_type == "activations":
                ov_component_per_head, ov_component = self.compute_ov_activations(
                    layer_value_activation, layer_output_projection
                )
            elif component_config.data_type == "weights":
                ov_component_per_head, ov_component = self.compute_ov_weights(
                    layer_value_projection, layer_output_projection
                )
            elif component_config.data_type == "gradients":
                ov_component_per_head, ov_component = self.compute_ov_gradients(
                    layer_value_gradient,
                    layer_output_gradient,
                    layer_value_projection,
                    layer_output_projection,
                )

            for head_idx, ov_component_head in ov_component_per_head.items():
                checkpoint_layer_component[
                    f"{_model_prefix}{layer_idx}.ov_circuit.{component_config.data_type}.heads.{head_idx}"
                ] = ov_component_head

            checkpoint_layer_component[
                f"{_model_prefix}{layer_idx}.ov_circuit.{component_config.data_type}"
            ] = ov_component

        return checkpoint_layer_component
