from metaworld_algorithms.config.utils import Initializer, StdType
from .nn import NeuralNetworkConfig, VanillaNetworkConfig
from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuousActionPolicyConfig:
    network_config: NeuralNetworkConfig = VanillaNetworkConfig(width=400, depth=3)
    """The config for the neural network to use for function approximation."""

    squash_tanh: bool = True
    """Whether or not to squash the outputs with tanh."""

    log_std_min: float | None = -20.0
    """The minimum possible log standard deviation for each action distribution."""

    log_std_max: float | None = 2.0
    """The maximum possible log standard deviation for each action distribution."""

    std_type: StdType = StdType.MLP_HEAD
    """How to learn the standard deviation of the distribution.
    `MLP_HEAD` means it will be an output head from the last layer of the MLP torso and therefore state-dependent.
    `PARAM` means it will be a learned parameter per action dimension that will be state-independent."""

    head_kernel_init: Initializer | None = None
    """Override the initializer to use for the MLP head weights."""

    head_bias_init: Initializer | None = None
    """Override the initializer to use for the MLP head biases."""


@dataclass(frozen=True)
class QValueFunctionConfig:
    network_config: NeuralNetworkConfig = VanillaNetworkConfig(width=400, depth=3)
    """The config for the neural network to use for function approximation."""

    use_classification: bool = False
    """Whether or not to use classification instead of regression."""


@dataclass(frozen=True)
class ValueFunctionConfig(QValueFunctionConfig): ...
