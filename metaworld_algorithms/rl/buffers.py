import abc

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from jaxtyping import Float
from typing import override

from metaworld_algorithms.types import (
    Action,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Rollout,
)


class AbstractReplayBuffer(abc.ABC):
    """Replay buffer for the single-task environments.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size,).
    When pushing samples to the buffer, the buffer accepts inputs of arbitrary batch dimensions.
    """

    obs: Float[Observation, " buffer_size"]
    actions: Float[Action, " buffer_size"]
    rewards: Float[npt.NDArray, "buffer_size 1"]
    next_obs: Float[Observation, " buffer_size"]
    dones: Float[npt.NDArray, "buffer_size 1"]
    pos: int

    @abc.abstractmethod
    def __init__(
        self,
        capacity: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def checkpoint(self) -> ReplayBufferCheckpoint: ...

    @abc.abstractmethod
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None: ...

    @abc.abstractmethod
    def add(
        self,
        obs: Observation,
        next_obs: Observation,
        action: Action,
        reward: Float[npt.NDArray, " *batch"],
        done: Float[npt.NDArray, " *batch"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        ...

    @abc.abstractmethod
    def sample(self, batch_size: int) -> ReplayBufferSamples: ...


class ReplayBuffer(AbstractReplayBuffer):
    """Replay buffer for the single-task environments.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size,).
    When pushing samples to the buffer, the buffer accepts inputs of arbitrary batch dimensions.
    """

    obs: Float[Observation, " buffer_size"]
    actions: Float[Action, " buffer_size"]
    rewards: Float[npt.NDArray, "buffer_size 1"]
    next_obs: Float[Observation, " buffer_size"]
    dones: Float[npt.NDArray, "buffer_size 1"]
    pos: int

    def __init__(
        self,
        capacity: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.full = False

        self.reset()  # Init buffer

    @override
    def reset(self):
        """Reinitialize the buffer."""
        self.obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self._action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0

    @override
    def checkpoint(self) -> ReplayBufferCheckpoint:
        return {
            "data": {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_obs": self.next_obs,
                "dones": self.dones,
                "pos": self.pos,
                "full": self.full,
            },
            "rng_state": self._rng.__getstate__(),
        }

    @override
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None:
        for key in ["data", "rng_state"]:
            assert key in ckpt

        for key in ["obs", "actions", "rewards", "next_obs", "dones", "pos", "full"]:
            assert key in ckpt["data"]
            setattr(self, key, ckpt["data"][key])

        self._rng.__setstate__(ckpt["rng_state"])

    @override
    def add(
        self,
        obs: Observation,
        next_obs: Observation,
        action: Action,
        reward: Float[npt.NDArray, " *batch"],
        done: Float[npt.NDArray, " *batch"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        if obs.ndim >= 2:
            assert (
                obs.shape[0] == action.shape[0] == reward.shape[0] == done.shape[0]
            ), "Batch size must be the same for all transition data."

            # Flatten any batch dims
            flat_obs = obs.reshape(-1, obs.shape[-1])
            flat_next_obs = next_obs.reshape(-1, next_obs.shape[-1])
            flat_action = action.reshape(-1, action.shape[-1])
            flat_reward = reward.reshape(
                -1, 1
            )  # Keep the last dim as 1 for consistency
            flat_done = done.reshape(-1, 1)  # Keep the last dim as 1 for consistency

            # Calculate number of new transitions
            n_transitions = len(flat_obs)

            # Handle buffer wraparound
            indices = np.arange(self.pos, self.pos + n_transitions) % self.capacity

            # Store the transitions
            self.obs[indices] = flat_obs
            self.next_obs[indices] = flat_next_obs
            self.actions[indices] = flat_action
            self.rewards[indices] = flat_reward
            self.dones[indices] = flat_done

            self.pos = (self.pos + n_transitions) % self.capacity
            if self.pos > self.capacity and not self.full:
                self.full = True
        else:
            self.obs[self.pos] = obs.copy()
            self.actions[self.pos] = action.copy()
            self.next_obs[self.pos] = next_obs.copy()
            self.dones[self.pos] = done.copy().reshape(-1, 1)
            self.rewards[self.pos] = reward.copy().reshape(-1, 1)

            self.pos += 1

        if self.pos > self.capacity and not self.full:
            self.full = True
        self.pos %= self.capacity

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos if not self.full else self.capacity, batch_size),
            size=(batch_size,),
        )

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )

        return ReplayBufferSamples(*batch)


class MultiTaskReplayBuffer(AbstractReplayBuffer):
    """Replay buffer for the multi-task benchmarks.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size, num_tasks,).
    When pushing samples to the buffer, the buffer only accepts inputs with batch shape (num_tasks,).
    """

    obs: Float[Observation, "buffer_size task"]
    actions: Float[Action, "buffer_size task"]
    rewards: Float[npt.NDArray, "buffer_size task 1"]
    next_obs: Float[Observation, "buffer_size task"]
    dones: Float[npt.NDArray, "buffer_size task 1"]
    pos: int

    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
        max_steps: int = 500,
    ) -> None:
        assert (
            total_capacity % num_tasks == 0
        ), "Total capacity must be divisible by the number of tasks."
        self.capacity = total_capacity // num_tasks
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.full = False

        # all needed for reward smoothing --> Reggie's original idea about scale and smoothness mattering
        self.max_steps = max_steps
        self.current_trajectory_start = 0

        self.reset(save_rewards=False)  # Init buffer

    @override
    def reset(self, save_rewards=False):
        """Reinitialize the buffer."""
        self.obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.num_tasks, self._action_shape), dtype=np.float32
        )
        self.rewards = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.next_obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.dones = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.pos = 0

        if save_rewards:
            self.org_rewards = np.zeros(
                (self.capacity, self.num_tasks, 1), dtype=np.float32
            )
            self.traj_start = 0

    @override
    def checkpoint(self) -> ReplayBufferCheckpoint:
        return {
            "data": {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_obs": self.next_obs,
                "dones": self.dones,
                "pos": self.pos,
                "full": self.full,
            },
            "rng_state": self._rng.__getstate__(),
        }

    @override
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None:
        for key in ["data", "rng_state"]:
            assert key in ckpt

        for key in ["obs", "actions", "rewards", "next_obs", "dones", "pos", "full"]:
            assert key in ckpt["data"]
            setattr(self, key, ckpt["data"][key])

        self._rng.__setstate__(ckpt["rng_state"])

    @override
    def add(
        self,
        obs: Float[Observation, " task"],
        next_obs: Float[Observation, " task"],
        action: Float[Action, " task"],
        reward: Float[npt.NDArray, " task"],
        done: Float[npt.NDArray, " task"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        # NOTE: assuming batch dim = task dim
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_tasks
        )

        self.obs[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.next_obs[self.pos] = next_obs.copy()
        self.dones[self.pos] = done.copy().reshape(-1, 1)
        self.rewards[self.pos] = reward.reshape(-1, 1).copy()

        self.pos = self.pos + 1
        if self.pos == self.capacity:
            self.full = True

        self.pos = self.pos % self.capacity

    def single_task_sample(self, task_idx: int, batch_size: int) -> ReplayBufferSamples:
        assert task_idx < self.num_tasks, "Task index out of bounds."

        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos if not self.full else self.capacity, batch_size),
            size=(batch_size,),
        )

        batch = (
            self.obs[sample_idx][task_idx],
            self.actions[sample_idx][task_idx],
            self.next_obs[sample_idx][task_idx],
            self.dones[sample_idx][task_idx],
            self.rewards[sample_idx][task_idx],
        )

        return ReplayBufferSamples(*batch)

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a batch of size `single_task_batch_size` for each task.

        Args:
            batch_size (int): The total batch size. Must be divisible by number of tasks

        Returns:
            ReplayBufferSamples: A batch of samples of batch shape (batch_size,).
        """
        assert (
            batch_size % self.num_tasks == 0
        ), "Batch size must be divisible by the number of tasks."
        single_task_batch_size = batch_size // self.num_tasks

        sample_idx = self._rng.integers(
            low=0,
            high=max(
                self.pos if not self.full else self.capacity, single_task_batch_size
            ),
            size=(single_task_batch_size,),
        )

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )

        mt_batch_size = single_task_batch_size * self.num_tasks
        batch = map(lambda x: x.reshape(mt_batch_size, *x.shape[2:]), batch)

        return ReplayBufferSamples(*batch)


def compute_gae(
    rollouts: Rollout,
    gamma: float,
    gae_lambda: float,
    last_values: Float[npt.NDArray, " task"],
    dones: Float[npt.NDArray, " task"],
) -> Rollout:
    assert rollouts.values is not None

    last_values = last_values.reshape(-1, 1)
    dones = dones.reshape(-1, 1)

    advantages = np.zeros_like(rollouts.rewards)

    # Adapted from https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py
    last_gae_lamda, returns = 0, None
    num_rollout_steps = rollouts.observations.shape[0]
    for timestep in reversed(range(num_rollout_steps)):
        if timestep == num_rollout_steps - 1:
            next_nonterminal = 1.0 - rollouts.dones
            next_values = last_values
        else:
            next_nonterminal = 1.0 - rollouts.dones[timestep + 1]
            next_values = rollouts.values[timestep + 1]
        delta = (
            rollouts.rewards[timestep]
            + next_nonterminal * gamma * next_values
            - rollouts.values[timestep]
        )
        advantages[timestep] = last_gae_lamda = (
            delta + next_nonterminal * gamma * gae_lambda * last_gae_lamda
        )
        returns = advantages + rollouts.values

    return rollouts._replace(
        returns=returns,
        advantages=advantages,
    )


class MultiTaskRolloutBuffer:
    num_rollout_steps: int
    num_tasks: int
    pos: int

    observations: Float[Observation, "task timestep"]
    actions: Float[Action, "task timestep"]
    rewards: Float[npt.NDArray, "task timestep 1"]
    dones: Float[npt.NDArray, "task timestep 1"]

    values: Float[npt.NDArray, "task timestep 1"]
    log_probs: Float[npt.NDArray, "task timestep 1"]
    means: Float[Action, "task timestep"]
    stds: Float[Action, "task timestep"]

    def __init__(
        self,
        num_rollout_steps: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None:
        self.num_rollout_steps = num_rollout_steps
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.reset()  # Init buffer

    def reset(self) -> None:
        """Reinitialize the buffer."""
        self.observations = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._action_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=np.float32
        )
        self.dones = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=np.float32
        )

        self.log_probs = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=np.float32
        )
        self.values = np.zeros_like(self.rewards)
        self.means = np.zeros_like(self.actions)
        self.stds = np.zeros_like(self.actions)
        self.pos = 0

    @property
    def ready(self) -> bool:
        return self.pos == self.num_rollout_steps

    def add(
        self,
        obs: Float[Observation, " task"],
        action: Float[Action, " task"],
        reward: Float[npt.NDArray, " task"],
        done: Float[npt.NDArray, " task"],
        value: Float[npt.NDArray, " task"] | None = None,
        log_prob: Float[npt.NDArray, " task"] | None = None,
        mean: Float[Action, " task"] | None = None,
        std: Float[Action, " task"] | None = None,
    ):
        # NOTE: assuming batch dim = task dim
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_tasks
        )

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy().reshape(-1, 1)
        self.dones[self.pos] = done.copy().reshape(-1, 1)

        if value is not None:
            self.values[self.pos] = value.copy()
        if log_prob is not None:
            self.log_probs[self.pos] = log_prob.reshape(-1, 1).copy()
        if mean is not None:
            self.means[self.pos] = mean.copy()
        if std is not None:
            self.stds[self.pos] = std.copy()

        self.pos += 1

    def get(
        self,
        compute_advantages: bool,
        last_values: Float[npt.NDArray, " task"] | None = None,
        dones: Float[npt.NDArray, " task"] | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
    ) -> Rollout:
        rollouts = Rollout(
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs,
            self.means,
            self.stds,
            self.values,
        )
        if compute_advantages:
            assert (
                last_values is not None
            ), "Must provide final value estimates if compute_advantages=True."
            assert (
                dones is not None
            ), "Must provide final value estimates if compute_advantages=True."
            assert not np.all(
                self.values == np.zeros_like(self.values)
            ), "Values must have been pushed to the buffer if compute_advantages=True."
            rollouts = compute_gae(rollouts, gamma, gae_lambda, last_values, dones)
            rollouts = rollouts._replace(
                episode_returns=rollouts.rewards.mean(axis=1).sum(axis=0).mean()
            )

        return rollouts
