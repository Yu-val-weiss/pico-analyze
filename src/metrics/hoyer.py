"""
Hoyer's sparsity metric is a measure of the sparsity of a matrix.
"""

import math

import torch

from src.config.base import BaseComponentConfig
from src.metrics._registry import register_metric
from src.metrics.base import BaseMetric


@register_metric("hoyer")
class HoyerMetric(BaseMetric):
    """
    Hoyer's sparsity metric is a measure of the sparsity of a matrix. Formally, it is defined as:

        Hoyer(P) = (sqrt(n) - ||P||_1 / ||P||_2) / (sqrt(n) - 1)

    where P is the parameter matrix, ||.||_1 is the L1 norm, and ||.||_2 is the L2 norm.
    """

    # NOTE: Any component is valid for the Hoyer metric.
    def validate_component(self, component_config: BaseComponentConfig) -> None: ...

    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Computes the Hoyer sparsity metric for a given component layer data.

        Args:
            component_layer_data: Tensor containing the data to analyze

        Returns:
            float: The computed Hoyer sparsity metric
        """

        x = component_layer_data.flatten()
        n = x.numel()

        # Compute the L1 and L2 norms of the component layer data
        l1_norm = torch.norm(component_layer_data, p=1).item()
        l2_norm = torch.norm(component_layer_data, p=2).item()
        if l2_norm == 0:
            return float("nan")
        return (math.sqrt(n) - l1_norm / l2_norm) / (math.sqrt(n) - 1)
