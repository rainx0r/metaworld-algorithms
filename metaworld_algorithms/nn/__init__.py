import flax.linen as nn

import mtrl.config.nn

from .base import MLP as VanillaNetwork
from .care import CARENetwork
from .film import FiLMNetwork
from .moore import MOORENetwork
from .multi_head import MultiHeadNetwork
from .paco import PaCoNetwork
from .soft_modules import SoftModularizationNetwork


def get_nn_arch_for_config(
    config: mtrl.config.nn.NeuralNetworkConfig,
) -> type[nn.Module]:
    if type(config) is mtrl.config.nn.MultiHeadConfig:
        return MultiHeadNetwork
    elif type(config) is mtrl.config.nn.SoftModulesConfig:
        return SoftModularizationNetwork
    elif type(config) is mtrl.config.nn.PaCoConfig:
        return PaCoNetwork
    elif type(config) is mtrl.config.nn.CAREConfig:
        return CARENetwork
    elif type(config) is mtrl.config.nn.FiLMConfig:
        return FiLMNetwork
    elif type(config) is mtrl.config.nn.MOOREConfig:
        return MOORENetwork
    elif isinstance(config, mtrl.config.nn.NeuralNetworkConfig):
        return VanillaNetwork


__all__ = ["VanillaNetwork", "MultiHeadNetwork", "SoftModularizationNetwork"]
