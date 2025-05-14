"""
Condition number is the ratio of the largest to smallest singular value of the input.
"""

import torch
from torch.linalg import svdvals

from src.config.base import BaseComponentConfig
from src.metrics._registry import register_metric
from src.metrics.base import BaseMetric


@register_metric("condition_number")
class ConditionNumberMetric(BaseMetric):
    """
    This metric computes the condition number of some component data. The condition number is the
    ratio of the largest to smallest singular value of the input. It gives a measure of how
    sensitive the output is to small changes in the input.
    """

    # NOTE: Any component is valid for the condition number metric.
    def validate_component(self, component_config: BaseComponentConfig) -> None: ...

    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Computes the condition number of the given input.

        Args:
            component_layer_data: Tensor containing the data to analyze

        Returns:
            float: The computed condition number
        """

        # Compute the singular values of the input
        component_layer_data = component_layer_data.to(torch.float32)

        try:
            singular_values: torch.Tensor = svdvals(component_layer_data)

            # Compute the condition number
            condition_number = singular_values.max() / singular_values.min()

            return condition_number.item()

        except Exception as e:
            print(f"Warning: SVD computation failed: {str(e)}")
            return float("inf")
