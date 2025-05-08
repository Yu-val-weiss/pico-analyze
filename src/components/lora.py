"""
Simple components are those that are a single layer. For example, the weight matrix of a layer is
a single component. In other words, simple components are those that can just be extracted directly
from the stored out checkpoint data without much additional computation.
"""

from typing import Any, Dict

import torch

from src.components._registry import register_component
from src.components.base_lora import BaseLoraComponent
from src.config.base import BaseComponentConfig
from src.utils.exceptions import InvalidComponentError


@register_component("lora")
class LoraComponent(BaseLoraComponent):
    """
    LoraComponent multiplies the lora parameters at each checkpoint.
    """

    def validate_component(self, component_config: BaseComponentConfig) -> None:
        """
        Lora components can only be weights.
        """
        if component_config.data_type != "weights":
            raise InvalidComponentError(
                f"Simple component only supports weights not {component_config.data_type}."
            )

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

        checkpoint_layer_component = {}

        _data = checkpoint_states[component_config.data_type]
        _model_prefix = self.get_model_prefix(checkpoint_states)

        for layer_idx in component_config.layers:
            layer_prefix = (
                f"{_model_prefix}{layer_idx}.{component_config.layer_suffixes}"
            )
            A_lora = _data[f"{layer_prefix}.A_lora"]
            B_lora = _data[f"{layer_prefix}.B_lora"]

            lora_component = B_lora @ A_lora

            checkpoint_layer_component[
                f"{_model_prefix}{layer_idx}.{component_config.layer_suffixes}.lora.{component_config.data_type}"
            ] = lora_component

            base_component = _data[layer_prefix]

            checkpoint_layer_component[
                f"{_model_prefix}{layer_idx}.{component_config.layer_suffixes}.base.{component_config.data_type}"
            ] = base_component

            full_component = base_component + lora_component * self.lora_s

            checkpoint_layer_component[
                f"{_model_prefix}{layer_idx}.{component_config.layer_suffixes}.full.{component_config.data_type}"
            ] = full_component

        return checkpoint_layer_component
