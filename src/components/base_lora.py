"""
Base class for components.
"""

from typing import Any, Dict

from src.components.base import BaseComponent


class BaseLoraComponent(BaseComponent):
    """
    Base class for lora components.
    Implements common initialisation.


    Components are functional objects that are used to generate a component from a given checkpoint
    state and a component configuration.
    """

    def __init__(self, training_config: Dict[str, Any]):
        super().__init__(training_config)

        relora_conf = training_config["model"].get("relora", None)
        if relora_conf is None:
            raise ValueError("must have 'relora' as a key in config")

        if relora_conf["trainable_scaling"]:
            raise NotImplementedError("not yet implemented for trainable scaling")

        self.lora_s = relora_conf["lora_alpha"] / relora_conf["r"]
