import abc
import time
from collections import deque
from typing import Deque, Generic, Self, TypeVar, override

import gymnasium as gym
import numpy as np
import orbax.checkpoint as ocp
import wandb
from flax import struct

from metaworld_algorithms.checkpoint import get_checkpoint_save_args
from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.config.rl import (
    AlgorithmConfig,
    OffPolicyTrainingConfig,
    OnPolicyTrainingConfig,
    TrainingConfig,
)
from metaworld_algorithms.rl.buffers import (
    AbstractReplayBuffer,
    MultiTaskRolloutBuffer,
)
from metaworld_algorithms.types import (
    Action,
    Agent,
    CheckpointMetadata,
    LogDict,
    LogProb,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Rollout,
    Value,
)

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)
DataType = TypeVar("DataType", ReplayBufferSamples, Rollout)


class Algorithm(
    abc.ABC,
    Agent,
    Generic[AlgorithmConfigType, TrainingConfigType, DataType],
    struct.PyTreeNode,
):
    """Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/agent.py"""

    num_tasks: int = struct.field(pytree_node=False)

    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: EnvConfig, seed: int = 1
    ) -> "Algorithm": ...

    @abc.abstractmethod
    def update(self, data: DataType) -> tuple[Self, LogDict]: ...

    @abc.abstractmethod
    def get_num_params(self) -> dict[str, int]: ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]: ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action: ...

    @abc.abstractmethod
    def train(
        self,
        config: TrainingConfigType,
        envs: gym.vector.VectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self: ...


class OffPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OffPolicyTrainingConfig, ReplayBufferSamples],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> AbstractReplayBuffer: ...

    @override
    def train(
        self,
        config: OffPolicyTrainingConfig,
        envs: gym.vector.VectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        has_autoreset = np.full((envs.num_envs,), False)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        replay_buffer = self.spawn_replay_buffer(env_config, config, seed)
        if buffer_checkpoint is not None:
            replay_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            if global_step < config.warmstart_steps:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                self, actions = self.sample_action(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            if not has_autoreset.any():
                replay_buffer.add(obs, next_obs, actions, rewards, terminations)
            elif has_autoreset.any() and not has_autoreset.all():
                # TODO: handle the case where only some envs have autoreset
                raise NotImplementedError(
                    "Only some envs resetting isn't implemented at the moment."
                )

            has_autoreset = np.logical_or(terminations, truncations)

            for i, env_ended in enumerate(has_autoreset):
                if env_ended:
                    global_episodic_return.append(infos["episode"]["r"][i])
                    global_episodic_length.append(infos["episode"]["l"][i])
                    episodes_ended += 1

            obs = next_obs

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    wandb.log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            if global_step > config.warmstart_steps:
                # Update the agent with data
                data = replay_buffer.sample(config.batch_size)
                self, logs = self.update(data)

                # Logging
                if global_step % 100 == 0:
                    sps_steps = (global_step - start_step) * envs.num_envs
                    sps = int(sps_steps / (time.time() - start_time))
                    print("SPS:", sps)

                    if track:
                        wandb.log({"charts/SPS": sps} | logs, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and has_autoreset.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        wandb.log(eval_metrics, step=total_steps)

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not has_autoreset.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                                buffer=replay_buffer,
                            ),
                            metrics={
                                k.removeprefix("charts/"): v
                                for k, v in eval_metrics.items()
                            },
                        )
        return self


class OnPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OnPolicyTrainingConfig, Rollout],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_dist_and_value(
        self, observation: Observation
    ) -> tuple[Self, Action, LogProb, Action, Action, Value]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: OnPolicyTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            training_config.rollout_steps,
            self.num_tasks,
            env_config.observation_space,
            env_config.action_space,
            seed,
        )

    @override
    def train(
        self,
        config: OnPolicyTrainingConfig,
        envs: gym.vector.VectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        has_autoreset = np.full((envs.num_envs,), False)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)
        # TODO:
        # if buffer_checkpoint is not None:
        #     rollout_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            self, actions, log_probs, means, stds, values = (
                self.sample_action_dist_and_value(obs)
            )

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            rollout_buffer.add(
                obs,
                actions,
                rewards,
                terminations or truncations,
                values,
                log_probs,
                means,
                stds,
            )

            has_autoreset = np.logical_or(terminations, truncations)
            for i, env_ended in enumerate(has_autoreset):
                if env_ended:
                    global_episodic_return.append(infos["episode"]["r"][i])
                    global_episodic_length.append(infos["episode"]["l"][i])
                    episodes_ended += 1

            obs = next_obs

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    wandb.log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            # Logging
            if global_step % 1_000 == 0:
                sps_steps = (global_step - start_step) * envs.num_envs
                sps = int(sps_steps / (time.time() - start_time))
                print("SPS:", sps)

                if track:
                    wandb.log({"charts/SPS": sps}, step=total_steps)

            if rollout_buffer.ready:
                last_values = None
                if config.compute_advantages:
                    self, _, _, _, _, last_values = self.sample_action_dist_and_value(
                        next_obs
                    )

                rollouts = rollout_buffer.get(
                    config.compute_advantages, last_values, terminations or truncations
                )

                # Flatten batch dims
                rollouts = Rollout(
                    *map(lambda x: x.reshape(-1, x.shape[-1]) if x else None, rollouts)  # pyright: ignore[reportArgumentType]
                )

                rollout_size = rollouts.observations.shape[0]
                minibatch_size = rollout_size // config.num_gradient_steps

                logs = {}
                batch_inds = np.arange(rollout_size)
                for epoch in range(config.num_epochs):
                    np.random.shuffle(batch_inds)
                    for start in range(0, rollout_size, minibatch_size):
                        end = start + minibatch_size
                        minibatch_rollout = Rollout(
                            *map(
                                lambda x: x[batch_inds[start:end]] if x else None,  # pyright: ignore[reportArgumentType]
                                rollouts,
                            )
                        )
                        self, logs = self.update(minibatch_rollout)

                    if config.target_kl is not None:
                        assert (
                            "losses/approx_kl" in logs
                        ), "Algorithm did not provide approximate KL div, but approx_kl is not None."
                        if logs["losses/approx_kl"] > config.target_kl:
                            print(
                                f"Stopped early at KL {logs['losses/approx_kl']}, ({epoch} epochs)"
                            )
                            break

                rollout_buffer.reset()

                if track:
                    wandb.log(logs, step=total_steps)

            # Evaluation
            if (
                config.evaluation_frequency > 0
                and episodes_ended % config.evaluation_frequency == 0
                and has_autoreset.any()
                and global_step > 0
            ):
                mean_success_rate, mean_returns, mean_success_per_task = (
                    env_config.evaluate(envs, self)
                )
                eval_metrics = {
                    "charts/mean_success_rate": float(mean_success_rate),
                    "charts/mean_evaluation_return": float(mean_returns),
                } | {
                    f"charts/{task_name}_success_rate": float(success_rate)
                    for task_name, success_rate in mean_success_per_task.items()
                }
                print(
                    f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                    + f" return: {mean_returns:.4f}"
                )

                if track:
                    wandb.log(eval_metrics, step=total_steps)

                # Reset envs again to exit eval mode
                obs, _ = envs.reset()

                # Checkpointing
                if checkpoint_manager is not None:
                    if not has_autoreset.all():
                        raise NotImplementedError(
                            "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                        )

                    checkpoint_manager.save(
                        total_steps,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                            # buffer=replay_buffer, TODO:
                        ),
                        metrics={
                            k.removeprefix("charts/"): v
                            for k, v in eval_metrics.items()
                        },
                    )

        return self
