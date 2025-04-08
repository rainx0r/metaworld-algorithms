# pyright: reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false, reportOptionalMemberAccess=false
from dataclasses import dataclass
from functools import cached_property
from typing import override

import gymnasium as gym
import numpy as np

from metaworld_algorithms.types import Agent, MetaLearningAgent, GymVectorEnv

from metaworld_algorithms.config.envs import EnvConfig, MetaLearningEnvConfig
from metaworld.evaluation import evaluation, metalearning_evaluation


@dataclass(frozen=True)
class MetaworldConfig(EnvConfig):
    reward_func_version: str = "v2"
    num_goals: int = 50
    reward_normalization_method: str | None = None
    env_name: str | None = None

    @cached_property
    @override
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            np.array([-1, -1, -1, -1], dtype=np.float32),
            np.array([+1, +1, +1, +1], dtype=np.float32),
        )

    @cached_property
    @override
    def observation_space(self) -> gym.Space:
        _HAND_SPACE = gym.spaces.Box(
            np.array([-0.525, 0.348, -0.0525]),
            np.array([+0.525, 1.025, 0.7]),
            dtype=np.float64,
        )

        goal_low = (-0.1, 0.85, 0.0)
        goal_high = (0.1, 0.9 + 1e-7, 0.0)

        goal_space = gym.spaces.Box(
            np.array(goal_low) + np.array([0, -0.083, 0.2499]),
            np.array(goal_high) + np.array([0, -0.083, 0.2501]),
            dtype=np.float64,
        )
        obs_obj_max_len = 14
        obj_low = np.full(obs_obj_max_len, -np.inf)
        obj_high = np.full(obs_obj_max_len, +np.inf)
        goal_low = goal_space.low
        goal_high = goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0

        env_obs_space = gym.spaces.Box(
            np.hstack(
                (
                    _HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    _HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    goal_low,
                )
            ),
            np.hstack(
                (
                    _HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    _HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    goal_high,
                )
            ),
            dtype=np.float64,
        )

        if self.use_one_hot:
            num_tasks = 1
            if self.env_id == "MT10":
                num_tasks = 10
            if self.env_id == "MT50":
                num_tasks = 50
            one_hot_ub = np.ones(num_tasks)
            one_hot_lb = np.zeros(num_tasks)

            env_obs_space = gym.spaces.Box(
                np.concatenate([env_obs_space.low, one_hot_lb]),
                np.concatenate([env_obs_space.high, one_hot_ub]),
                dtype=np.float64,
            )

        return env_obs_space

    @override
    def evaluate(
        self, envs: GymVectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        return evaluation(agent, envs, num_episodes=self.evaluation_num_episodes)[:3]

    @override
    def spawn(self, seed: int = 1) -> GymVectorEnv:
        return gym.make_vec(  # pyright: ignore[reportReturnType]
            f"Meta-World/{self.env_id}",
            seed=seed,
            use_one_hot=self.use_one_hot,
            terminate_on_success=self.terminate_on_success,
            vector_strategy="async",
            reward_function_version=self.reward_func_version,
            num_goals=self.num_goals,
            reward_normalization_method=self.reward_normalization_method,
        )


@dataclass(frozen=True)
class MetaworldMetaLearningConfig(MetaworldConfig, MetaLearningEnvConfig):
    use_one_hot: bool = False
    meta_batch_size: int = 20

    total_goals_per_task_train: int = 50
    total_goals_per_task_test: int = 40

    evaluation_num_episodes: int = 3
    evaluation_adaptation_steps: int = 1
    evaluation_adaptation_episodes: int = 10

    @override
    def evaluate_metalearning(
        self, envs: GymVectorEnv, agent: MetaLearningAgent
    ) -> tuple[float, float, dict[str, float]]:
        # NOTE: "agent" here is the same interface as what Metaworld expects
        # but, because of `Rollout` being a local class, the type checker can't fully certify that
        # We could just use Metaworld's types throughout the project, but
        # I kind of don't want to rely on `from metaworld` imports outside this file.

        if self.env_id == "ML10" or self.env_id == "ML45":
            num_classes = 5
        elif self.env_id.startswith("ML1-"):
            num_classes = 1
        else:
            raise NotImplementedError(f"Unknown env_id: {self.env_id}")

        num_evals = (num_classes * self.total_goals_per_task_test) // self.meta_batch_size

        return metalearning_evaluation(
            agent,  # pyright: ignore[reportArgumentType]
            envs,
            evaluation_episodes=self.evaluation_num_episodes,
            max_episode_steps=self.max_episode_steps,
            adaptation_steps=self.evaluation_adaptation_steps,
            adaptation_episodes=self.evaluation_adaptation_episodes,
            num_evals=num_evals,
        )

    @override
    def evaluate_metalearning_on_train(
        self, envs: GymVectorEnv, agent: MetaLearningAgent
    ) -> tuple[float, float, dict[str, float]]:
        if self.env_id == "ML10":
            num_classes = 10
        elif  self.env_id == "ML45":
            num_classes = 45
        elif self.env_id.startswith("ML1-"):
            num_classes = 1
        else:
            raise NotImplementedError(f"Unknown env_id: {self.env_id}")

        num_evals = (num_classes * self.total_goals_per_task_train) // self.meta_batch_size

        return metalearning_evaluation(
            agent,  # pyright: ignore[reportArgumentType]
            envs,
            evaluation_episodes=self.evaluation_num_episodes,
            max_episode_steps=self.max_episode_steps,
            adaptation_steps=self.evaluation_adaptation_steps,
            adaptation_episodes=self.evaluation_adaptation_episodes,
            num_evals=num_evals,
        )

    @override
    def spawn(self, seed: int = 1) -> GymVectorEnv:
        kwargs = dict(
            seed=seed,
            terminate_on_success=self.terminate_on_success,
            vector_strategy="async",
            meta_batch_size=self.meta_batch_size,
            total_tasks_per_cls=self.total_goals_per_task_train,
            reward_function_version=self.reward_func_version,
        )
        if self.env_name:
            kwargs["env_name"] = self.env_name
        return gym.make_vec(  # pyright: ignore[reportReturnType]
            f"Meta-World/{self.env_id}-train",
            **kwargs, # pyright: ignore[reportArgumentType]
        )

    @override
    def spawn_test(self, seed: int = 1) -> GymVectorEnv:
        kwargs = dict(
            seed=seed,
            terminate_on_success=True,
            vector_strategy="async",
            meta_batch_size=self.meta_batch_size,
            total_tasks_per_cls=self.total_goals_per_task_test,
            reward_function_version=self.reward_func_version,
        )
        if self.env_name:
            kwargs["env_name"] = self.env_name
        return gym.make_vec(  # pyright: ignore[reportReturnType]
            f"Meta-World/{self.env_id}-test",
            **kwargs, # pyright: ignore[reportArgumentType]
        )
