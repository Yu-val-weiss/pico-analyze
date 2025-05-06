"""
PER metric computes the Proportional Effective Rank (PER) of activations or gradients.
"""

import torch

from src.config.base import BaseComponentConfig
from src.metrics._registry import register_metric
from src.metrics.base import BaseMetric
from src.utils.exceptions import InvalidComponentError


@register_metric("per")
class PERMetric(BaseMetric):
    """
    Compute the Proportional Effective Rank (PER) of some component data. The PER is defined as the
    entropy over the normalised singular values of a given matrix.

    Formally, if we let S = [s_1, ..., s_n] be the singular values of a parameter matrix P, then the PER is given by:

        PER(P) = - sum(s_i / sum(s)) * log2(s_i / sum(s)) for i = 1 to n

    where s = sum(s) is the sum of the singular values.
    """

    def validate_component(self, component_config: BaseComponentConfig) -> None:
        """
        NOTE: The PER metric is only valid for weights and gradients.
        """
        if component_config.data_type not in ["weights", "gradients"]:
            raise InvalidComponentError(
                f"PER metric only supports weights and gradients, not {component_config.data_type} "
                f"(component: {component_config.component_name})."
            )

    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Computes the PER of a given layer.

        Args:
            component_layer_data: Tensor containing the data to analyze

        Returns:
            float: The computed PER
        """

        component_layer_data = component_layer_data.to(torch.float32)
        layer_singular_values = torch.svd(component_layer_data).S

        # standardize singular values
        layer_singular_values = layer_singular_values / layer_singular_values.sum()

        # compute effective rank (ER) and proportional effective rank (PER)
        layer_er = torch.exp(
            -torch.sum(layer_singular_values * torch.log(layer_singular_values))
        ).item()
        layer_per = layer_er / len(layer_singular_values)

        return layer_per
