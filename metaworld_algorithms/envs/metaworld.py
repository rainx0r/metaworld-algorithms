# pyright: reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false, reportOptionalMemberAccess=false
from dataclasses import dataclass
from functools import cached_property
from typing import override

import gymnasium as gym
import numpy as np

from metaworld_algorithms.types import Agent

from metaworld_algorithms.config.envs import EnvConfig
from metaworld.evaluation import evaluation


@dataclass(frozen=True)
class MetaworldConfig(EnvConfig):
    reward_func_version: str = "v2"
    num_goals: int = 50
    reward_normalization_method: str | None = None

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
        self, envs: gym.vector.VectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        assert isinstance(envs, gym.vector.AsyncVectorEnv) or isinstance(
            envs, gym.vector.SyncVectorEnv
        )
        return evaluation(agent, envs, num_episodes=self.evaluation_num_episodes)

    @override
    def spawn(self, seed: int = 1) -> gym.vector.VectorEnv:
        return gym.make_vec(
            f"Meta-World/{self.env_id}",
            seed=seed,
            use_one_hot=self.use_one_hot,
            terminate_on_success=self.terminate_on_success,
            vector_strategy="async",
            reward_function_version=self.reward_func_version,
            num_goals=self.num_goals,
            reward_normalization_method=self.reward_normalization_method,
        )
