from .nn import NeuralNetworkConfig
from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuousActionPolicyConfig:
    network_config: NeuralNetworkConfig = NeuralNetworkConfig(width=400, depth=3)
    """The config for the neural network to use for function approximation."""

    squash_tanh: bool = True
    """Whether or not to squash the outputs with tanh."""

    log_std_min: float = -20.0
    """The minimum possible log standard deviation for each action distribution."""

    log_std_max: float = 2.0
    """The maximum possible log standard deviation for each action distribution."""


@dataclass(frozen=True)
class QValueFunctionConfig:
    network_config: NeuralNetworkConfig = NeuralNetworkConfig(width=400, depth=3)
    """The config for the neural network to use for function approximation."""

    use_classification: bool = False
    """Whether or not to use classification instead of regression."""

@dataclass(frozen=True)
class ValueFunctionConfig(QValueFunctionConfig): ...
