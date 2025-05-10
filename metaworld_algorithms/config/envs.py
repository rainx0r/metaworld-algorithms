import abc
from dataclasses import dataclass
from functools import cached_property

import gymnasium as gym

from metaworld_algorithms.types import Agent, GymVectorEnv, MetaLearningAgent


@dataclass(frozen=True)
class EnvConfig(abc.ABC):
    env_id: str
    use_one_hot: bool = True
    max_episode_steps: int = 500
    evaluation_num_episodes: int = 50
    terminate_on_success: bool = False

    @cached_property
    @abc.abstractmethod
    def action_space(self) -> gym.Space: ...

    @cached_property
    @abc.abstractmethod
    def observation_space(self) -> gym.Space: ...

    @abc.abstractmethod
    def spawn(self, seed: int = 1) -> GymVectorEnv: ...

    @abc.abstractmethod
    def evaluate(
        self, envs: GymVectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]: ...


@dataclass(frozen=True)
class MetaLearningEnvConfig(EnvConfig):
    recurrent_info_in_obs: bool = False

    @abc.abstractmethod
    def spawn_test(self, seed: int = 1) -> GymVectorEnv: ...

    @abc.abstractmethod
    def evaluate_metalearning(
        self, envs: GymVectorEnv, agent: MetaLearningAgent
    ) -> tuple[float, float, dict[str, float]]: ...

    @abc.abstractmethod
    def evaluate_metalearning_on_train(
        self, envs: GymVectorEnv, agent: MetaLearningAgent
    ) -> tuple[float, float, dict[str, float]]: ...
