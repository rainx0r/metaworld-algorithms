from metaworld_algorithms.config.rl import AlgorithmConfig

from .base import Algorithm, OffPolicyAlgorithm
from .mtsac import MTSAC, MTSACConfig
from .mtppo import MTPPOConfig, MTPPO


def get_algorithm_for_config(config: AlgorithmConfig) -> type[Algorithm]:
    if type(config) is MTSACConfig:
        return MTSAC
    elif type(config) is MTPPOConfig:
        return MTPPO
    else:
        raise ValueError(f"Invalid config type: {type(config)}")


__all__ = [
    "Algorithm",
    "OffPolicyAlgorithm",
    "MTSAC",
    "MTPPO",
    "MTSACConfig",
    "MTPPOConfig",
]
