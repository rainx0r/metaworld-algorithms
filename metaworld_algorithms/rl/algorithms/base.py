import abc
import time
from collections import deque
from typing import Deque, Generic, Self, TypeVar, override

import numpy as np
import orbax.checkpoint as ocp
import wandb
from flax import struct

from metaworld_algorithms.checkpoint import get_checkpoint_save_args
from metaworld_algorithms.config.envs import EnvConfig, MetaLearningEnvConfig
from metaworld_algorithms.config.rl import (
    AlgorithmConfig,
    GradientBasedMetaLearningTrainingConfig,
    MetaLearningTrainingConfig,
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
    GymVectorEnv,
    LogDict,
    LogProb,
    MetaLearningAgent,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Rollout,
    Value,
)

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)
EnvConfigType = TypeVar("EnvConfigType", bound=EnvConfig)
MetaLearningTrainingConfigType = TypeVar(
    "MetaLearningTrainingConfigType", bound=MetaLearningTrainingConfig
)
DataType = TypeVar("DataType", ReplayBufferSamples, Rollout, list[Rollout])


class Algorithm(
    abc.ABC,
    Agent,
    Generic[AlgorithmConfigType, TrainingConfigType, EnvConfigType, DataType],
    struct.PyTreeNode,
):
    """Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/agent.py"""

    num_tasks: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)

    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: EnvConfigType, seed: int = 1
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
        envs: GymVectorEnv,
        env_config: EnvConfigType,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self: ...


class MetaLearningAlgorithm(
    Algorithm[
        AlgorithmConfigType,
        MetaLearningTrainingConfigType,
        MetaLearningEnvConfig,
        DataType,
    ],
    Generic[AlgorithmConfigType, MetaLearningTrainingConfigType, DataType],
):
    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: MetaLearningEnvConfig, seed: int = 1
    ) -> "MetaLearningAlgorithm": ...

    @abc.abstractmethod
    def wrap(self) -> MetaLearningAgent: ...

    @abc.abstractmethod
    def train(
        self,
        config: MetaLearningTrainingConfigType,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self: ...


class GradientBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, GradientBasedMetaLearningTrainingConfig, list[Rollout]
    ],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_dist_and_value(
        self, observation: Observation
    ) -> tuple[Self, Action, LogProb, Action, Action, Value]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: GradientBasedMetaLearningTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @abc.abstractmethod
    def adapt(self, rollouts: Rollout) -> Self: ...

    @abc.abstractmethod
    def init_ensemble_networks(self) -> Self: ...

    @override
    def train(
        self,
        config: GradientBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        obs, _ = zip(*envs.call("sample_tasks"))
        obs = np.stack(obs)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
        )

        truncated = np.full((envs.num_envs,), False)
        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")
            self = self.init_ensemble_networks()
            all_rollouts: list[Rollout] = []

            # Sampling step
            # Collect num_inner_gradient_steps D datasets + collect 1 D' dataset
            for _step in range(config.num_inner_gradient_steps + 1):
                print(f"- Collecting inner step {_step}")
                while not rollout_buffer.ready:
                    self, actions, log_probs, means, stds, values = (
                        self.sample_action_dist_and_value(obs)
                    )

                    next_obs, reward, _, truncated, _ = envs.step(actions)
                    rollout_buffer.add(
                        obs, actions, reward, truncated, values, log_probs, means, stds
                    )
                    obs = next_obs

                self, _, _, _, _, last_values = self.sample_action_dist_and_value(obs)

                rollouts = rollout_buffer.get(
                    compute_advantages=True,
                    compute_episode_returns=True,
                    gamma=self.gamma,
                    gae_lambda=config.gae_lambda,
                    last_values=last_values,
                    dones=truncated,
                    # TODO: maybe we should bring LinearFeatureBaseline / advantage norm back?
                    # fit_baseline=config.fit_baseline,
                    # normalize_advantages=True,
                )
                all_rollouts.append(rollouts)
                rollout_buffer.reset()

                # Inner policy update for the sake of sampling close to adapted policy during the
                # computation of the objective.
                if _step < config.num_inner_gradient_steps:
                    print(f"- Adaptation step {_step}")
                    self = self.adapt(rollouts)

            assert all_rollouts[-1].episode_returns is not None
            mean_episodic_return = all_rollouts[-1].episode_returns
            print("- Mean episodic return: ", mean_episodic_return)
            if track:
                wandb.log(
                    {"charts/mean_episodic_returns": mean_episodic_return},
                    step=global_step,
                )

            # Outer policy update
            print("- Computing outer step")
            self, logs = self.update(all_rollouts)

            # Evaluation
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")
                eval_success_rate, eval_mean_return, eval_success_rate_per_task = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                logs["charts/mean_success_rate"] = float(eval_success_rate)
                logs["charts/mean_evaluation_return"] = float(eval_mean_return)
                for task_name, success_rate in eval_success_rate_per_task.items():
                    logs[f"charts/{task_name}_success_rate"] = float(success_rate)

                if config.evaluate_on_train:
                    print("- Evaluating on the train set...")
                    # num_evals = (
                    #     len(benchmark.train_classes) * config.num_evaluation_goals
                    # ) // config.meta_batch_size
                    _, _, eval_success_rate_per_train_task = (
                        env_config.evaluate_metalearning(
                            envs=envs,
                            agent=self.wrap(),
                        )
                    )
                    for (
                        task_name,
                        success_rate,
                    ) in eval_success_rate_per_train_task.items():
                        logs[f"charts/{task_name}_train_success_rate"] = float(
                            success_rate
                        )

                    envs.call("toggle_terminate_on_success", False)

                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self, envs, global_step, episodes_ended, run_timestamp
                        ),
                        metrics={k.removeprefix("charts/"): v for k, v in logs.items()},
                    )
                    print("- Saved Model")

            # Logging
            print(logs)
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                wandb.log({"charts/SPS": sps} | logs, step=global_step)

            # Set tasks for next iteration
            obs, _ = zip(*envs.call("sample_tasks"))
            obs = np.stack(obs)

        return self


class OffPolicyAlgorithm(
    Algorithm[
        AlgorithmConfigType, OffPolicyTrainingConfig, EnvConfig, ReplayBufferSamples
    ],
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
        envs: GymVectorEnv,
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
                        env_config.evaluate(envs, self)[:3]
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
    Algorithm[AlgorithmConfigType, OnPolicyTrainingConfig, EnvConfig, Rollout],
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
        envs: GymVectorEnv,
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

        start_time = time.time()

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            self, actions, log_probs, means, stds, values = (
                self.sample_action_dist_and_value(obs)
            )

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            if not has_autoreset.any():
                rollout_buffer.add(
                    obs,
                    actions,
                    rewards,
                    has_autoreset.astype(np.float32),
                    values,
                    log_probs,
                    means,
                    stds,
                )
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
                    config.compute_advantages,
                    last_values=last_values,
                    dones=has_autoreset.astype(np.float32),
                )

                # Flatten batch dims
                rollouts = Rollout(
                    *map(lambda x: x.reshape(-1, x.shape[-1]) if x is not None else None, rollouts)  # pyright: ignore[reportArgumentType]
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
                                lambda x: x[batch_inds[start:end]] if x is not None else None,  # pyright: ignore[reportArgumentType]
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
                        env_config.evaluate(envs, self)[:3]
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
                            ),
                            metrics={
                                k.removeprefix("charts/"): v
                                for k, v in eval_metrics.items()
                            },
                        )

        return self
