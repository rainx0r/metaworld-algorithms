from metaworld_algorithms.config.rl import AlgorithmConfig

from .base import Algorithm, OffPolicyAlgorithm
from .mtsac import MTSAC, MTSACConfig
from .ppo import PPOConfig, PPO
from .sac import SAC, SACConfig
from .mamltrpo import MAMLTRPO, MAMLTRPOConfig


def get_algorithm_for_config(config: AlgorithmConfig) -> type[Algorithm]:
    if type(config) is MTSACConfig:
        return MTSAC
    elif type(config) is PPOConfig:
        return PPO
    elif type(config) is SACConfig:
        return SAC
    elif type(config) is MAMLTRPOConfig:
        return MAMLTRPO
    else:
        raise ValueError(f"Invalid config type: {type(config)}")


__all__ = [
    "Algorithm",
    "OffPolicyAlgorithm",
    "MTSAC",
    "PPO",
    "MTSACConfig",
    "PPOConfig",
]
