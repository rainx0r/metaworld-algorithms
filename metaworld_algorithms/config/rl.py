from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaworld_algorithms.rl.algorithms.base import Algorithm

    from .envs import EnvConfig


@dataclass(frozen=True)
class AlgorithmConfig:
    num_tasks: int
    gamma: float = 0.99

    def spawn(self, env: "EnvConfig", seed: int) -> "Algorithm":
        from metaworld_algorithms.rl.algorithms import get_algorithm_for_config

        return get_algorithm_for_config(self).initialize(self, env, seed)


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    total_steps: int
    evaluation_frequency: int = 200_000 // 500
    """Evaluation frequency in total environment episodes."""


@dataclass(frozen=True)
class OffPolicyTrainingConfig(TrainingConfig):
    warmstart_steps: int = int(4e3)
    buffer_size: int = int(1e6)
    batch_size: int = 1280


@dataclass(frozen=True)
class OnPolicyTrainingConfig(TrainingConfig):
    rollout_steps: int = 10_000


@dataclass(frozen=True)
class MetaLearningTrainingConfig(TrainingConfig):
    meta_batch_size: int = 20
    rollouts_per_task: int = 10
    evaluate_on_train: bool = False

    evaluation_frequency: int = 1_000_000
    """Evaluation frequency in total environment timesteps."""


@dataclass(frozen=True)
class GradientBasedMetaLearningTrainingConfig(MetaLearningTrainingConfig):
    num_inner_gradient_steps: int = 1
