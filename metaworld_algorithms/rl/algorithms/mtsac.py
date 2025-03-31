"""Inspired by https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""

import dataclasses
from functools import partial
from typing import Self, override

import flax.linen as nn
import gymnasium as gym
import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from flax import struct
from flax.core import FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import AlgorithmConfig, OffPolicyTrainingConfig
from metaworld_algorithms.optim.pcgrad import PCGradState
from metaworld_algorithms.rl.buffers import MultiTaskReplayBuffer
from metaworld_algorithms.rl.networks import (
    ContinuousActionPolicy,
    Ensemble,
    QValueFunction,
)
from metaworld_algorithms.types import (
    Action,
    LogDict,
    Observation,
    ReplayBufferSamples,
)

from .base import OffPolicyAlgorithm
from .utils import TrainState


class MultiTaskTemperature(nn.Module):
    num_tasks: int
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full(
                (self.num_tasks,), jnp.log(self.initial_temperature)
            ),
        )

    def __call__(
        self, task_ids: Float[Array, "... num_tasks"]
    ) -> Float[Array, "... 1"]:
        return jnp.exp(task_ids @ self.log_alpha.reshape(-1, 1))


class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None


@jax.jit
def _sample_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action(
    actor: TrainState, observation: Observation
) -> Float[Array, "... action_dim"]:
    return actor.apply_fn(actor.params, observation).mode()


def extract_task_weights(
    alpha_params: FrozenDict, task_ids: Float[np.ndarray, "... num_tasks"]
) -> Float[Array, "... 1"]:
    log_alpha: jax.Array
    task_weights: jax.Array

    log_alpha = alpha_params["params"]["log_alpha"]  # pyright: ignore [reportAssignmentType]
    task_weights = jax.nn.softmax(-log_alpha)
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_alpha.shape[0]
    return task_weights


@dataclasses.dataclass(frozen=True)
class MTSACConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False
    max_q_value: float | None = 5000


class MTSAC(OffPolicyAlgorithm[MTSACConfig]):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    use_task_weights: bool = struct.field(pytree_node=False)
    split_actor_losses: bool = struct.field(pytree_node=False)
    split_critic_losses: bool = struct.field(pytree_node=False)
    num_critics: int = struct.field(pytree_node=False)
    max_q_value: float | None = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: MTSACConfig, env_config: EnvConfig, seed: int = 1
    ) -> "MTSAC":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, critic_init_key, alpha_init_key = (
            jax.random.split(master_key, 4)
        )

        actor_net = ContinuousActionPolicy(
            int(np.prod(env_config.action_space.shape)), config=config.actor_config
        )
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_init_key, dummy_obs),
            tx=config.actor_config.network_config.optimizer.spawn(),
        )

        print("Actor Arch:", jax.tree_util.tree_map(jnp.shape, actor.params))
        print("Actor Params:", sum(x.size for x in jax.tree.leaves(actor.params)))

        critic_cls = partial(QValueFunction, config=config.critic_config)
        critic_net = Ensemble(critic_cls, num=config.num_critics)
        dummy_action = jnp.array(
            [env_config.action_space.sample() for _ in range(config.num_tasks)]
        )
        critic_init_params = critic_net.init(critic_init_key, dummy_obs, dummy_action)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init_params,
            target_params=critic_init_params,
            tx=config.critic_config.network_config.optimizer.spawn(),
        )

        print("Critic Arch:", jax.tree_util.tree_map(jnp.shape, critic.params))
        print("Critic Params:", sum(x.size for x in jax.tree.leaves(critic.params)))

        alpha_net = MultiTaskTemperature(config.num_tasks, config.initial_temperature)
        dummy_task_ids = jnp.array(
            [np.ones((config.num_tasks,)) for _ in range(config.num_tasks)]
        )
        alpha = TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(alpha_init_key, dummy_task_ids),
            tx=config.temperature_optimizer_config.spawn(),
        )

        target_entropy = -np.prod(env_config.action_space.shape).item()

        return MTSAC(
            num_tasks=config.num_tasks,
            actor=actor,
            critic=critic,
            alpha=alpha,
            key=algorithm_key,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            use_task_weights=config.use_task_weights,
            num_critics=config.num_critics,
            split_actor_losses=config.actor_config.network_config.optimizer.requires_split_task_losses,
            split_critic_losses=config.critic_config.network_config.optimizer.requires_split_task_losses,
            max_q_value=config.max_q_value,
        )

    @override
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> MultiTaskReplayBuffer:
        return MultiTaskReplayBuffer(
            total_capacity=config.buffer_size,
            num_tasks=self.num_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "actor_num_params": sum(x.size for x in jax.tree.leaves(self.actor.params)),
            "critic_num_params": sum(
                x.size for x in jax.tree.leaves(self.critic.params)
            ),
        }

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.actor, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observations: Observation) -> Action:
        return jax.device_get(_eval_action(self.actor, observations))

    def split_data_by_tasks(
        self,
        data: PyTree[Float[Array, "batch data_dim"]],
        task_ids: Float[npt.NDArray, "batch num_tasks"],
    ) -> PyTree[Float[Array, "num_tasks per_task_batch data_dim"]]:
        tasks = jnp.argmax(task_ids, axis=1)
        sorted_indices = jnp.argsort(tasks)

        def group_by_task_leaf(
            leaf: Float[Array, "batch data_dim"],
        ) -> Float[Array, "task task_batch data_dim"]:
            leaf_sorted = leaf[sorted_indices]
            return leaf_sorted.reshape(self.num_tasks, -1, leaf.shape[1])

        return jax.tree.map(group_by_task_leaf, data), sorted_indices

    def unsplit_data_by_tasks(
        self,
        split_data: PyTree[Float[Array, "num_tasks per_task_batch data_dim"]],
        sort_indices: jax.Array,
    ) -> PyTree[Float[Array, "batch data_dim"]]:
        def reconstruct_leaf(
            leaf: Float[Array, "num_tasks per_task_batch data_dim"],
        ) -> Float[Array, "batch data_dim"]:
            batch_size = leaf.shape[0] * leaf.shape[1]
            flat = leaf.reshape(batch_size, leaf.shape[-1])
            # Create inverse permutation
            inverse_indices = jnp.zeros_like(sort_indices)
            inverse_indices = inverse_indices.at[sort_indices].set(
                jnp.arange(batch_size)
            )
            return flat[inverse_indices]

        return jax.tree.map(reconstruct_leaf, split_data)

    def update_critic(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "*batch 1"],
        task_weights: Float[Array, "*batch 1"] | None = None,
    ) -> tuple[Self, LogDict]:
        key, critic_loss_key = jax.random.split(self.key)

        # Sample a'
        if self.split_critic_losses:
            next_actions, next_action_log_probs = jax.vmap(
                lambda x: self.actor.apply_fn(self.actor.params, x).sample_and_log_prob(
                    seed=critic_loss_key
                )
            )(data.observations)
            q_values = jax.vmap(self.critic.apply_fn, in_axes=(None, 0, 0))(
                self.critic.target_params, data.next_observations, next_actions
            )
        else:
            next_actions, next_action_log_probs = self.actor.apply_fn(
                self.actor.params, data.next_observations
            ).sample_and_log_prob(seed=critic_loss_key)
            q_values = self.critic.apply_fn(
                self.critic.target_params, data.next_observations, next_actions
            )

        def critic_loss(
            params: FrozenDict,
            _data: ReplayBufferSamples,
            _q_values: Float[Array, "#batch 1"],
            _alpha_val: Float[Array, "#batch 1"],
            _next_action_log_probs: Float[Array, " #batch"],
            _task_weights: Float[Array, "#batch 1"] | None = None,
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            # next_action_log_probs is (B,) shaped because of the sum(axis=1), while Q values are (B, 1)
            min_qf_next_target = jnp.min(
                _q_values, axis=0
            ) - _alpha_val * _next_action_log_probs.reshape(-1, 1)

            next_q_value = jax.lax.stop_gradient(
                _data.rewards + (1 - _data.dones) * self.gamma * min_qf_next_target
            )

            q_pred = self.critic.apply_fn(params, _data.observations, _data.actions)

            if self.max_q_value is not None:
                # HACK: Clipping Q values to approximate theoretical maximum for Metaworld
                next_q_value = jnp.clip(
                    next_q_value, -self.max_q_value, self.max_q_value
                )
                q_pred = jnp.clip(q_pred, -self.max_q_value, self.max_q_value)

            if _task_weights is not None:
                loss = (_task_weights * (q_pred - next_q_value) ** 2).mean()
            else:
                loss = ((q_pred - next_q_value) ** 2).mean()
            return loss, q_pred.mean()

        if self.split_critic_losses:
            (critic_loss_value, qf_values), critic_grads = jax.vmap(
                jax.value_and_grad(critic_loss, has_aux=True),
                in_axes=(None, 0, 0, 0, 0, 0),
                out_axes=0,
            )(
                self.critic.params,
                data,
                q_values,
                alpha_val,
                next_action_log_probs,
                task_weights,
            )
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), critic_grads)
            )
        else:
            (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
                critic_loss, has_aux=True
            )(
                self.critic.params,
                data,
                q_values,
                alpha_val,
                next_action_log_probs,
                task_weights,
            )
            flat_grads, _ = flatten_util.ravel_pytree(critic_grads)

        key, optimizer_key = jax.random.split(key)
        critic = self.critic.apply_gradients(
            grads=critic_grads,
            optimizer_extra_args={
                "task_losses": critic_loss_value,
                "key": optimizer_key,
            },
        )
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params,
                critic.target_params,  # pyright: ignore [reportArgumentType]
                self.tau,
            )
        )
        flat_params_crit, _ = flatten_util.ravel_pytree(critic.params)

        return self.replace(critic=critic, key=key), {
            "losses/qf_values": qf_values.mean(),
            "losses/qf_loss": critic_loss_value.mean(),
            "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/critic_params_norm": jnp.linalg.norm(flat_params_crit),
        }

    def update_actor(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "batch 1"],
        task_weights: Float[Array, "batch 1"] | None = None,
    ) -> tuple[Self, Float[Array, " batch"], LogDict]:
        key, actor_loss_key = jax.random.split(self.key)

        def actor_loss(
            params: FrozenDict,
            _data: ReplayBufferSamples,
            _alpha_val: Float[Array, "batch 1"],
            _task_weights: Float[Array, "batch 1"] | None = None,
        ):
            action_samples, log_probs = self.actor.apply_fn(
                params, _data.observations
            ).sample_and_log_prob(seed=actor_loss_key)
            log_probs = log_probs.reshape(-1, 1)

            q_values = self.critic.apply_fn(
                self.critic.params, _data.observations, action_samples
            )
            min_qf_values = jnp.min(q_values, axis=0)
            if _task_weights is not None:
                loss = (task_weights * (_alpha_val * log_probs - min_qf_values)).mean()
            else:
                loss = (_alpha_val * log_probs - min_qf_values).mean()
            return loss, log_probs

        if self.split_actor_losses:
            (actor_loss_value, log_probs), actor_grads = jax.vmap(
                jax.value_and_grad(actor_loss, has_aux=True),
                in_axes=(None, 0, 0, 0),
                out_axes=0,
            )(self.actor.params, data, alpha_val, task_weights)
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), actor_grads)
            )
        else:
            (actor_loss_value, log_probs), actor_grads = jax.value_and_grad(
                actor_loss, has_aux=True
            )(self.actor.params, data, alpha_val, task_weights)
            flat_grads, _ = flatten_util.ravel_pytree(actor_grads)

        key, optimizer_key = jax.random.split(key)
        actor = self.actor.apply_gradients(
            grads=actor_grads,
            optimizer_extra_args={
                "task_losses": actor_loss_value,
                "key": optimizer_key,
            },
        )

        flat_params_act, _ = flatten_util.ravel_pytree(actor.params)
        logs = {
            "losses/actor_loss": actor_loss_value.mean(),
            "metrics/actor_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/actor_params_norm": jnp.linalg.norm(flat_params_act),
        }

        return (self.replace(actor=actor, key=key), log_probs, logs)

    def update_alpha(
        self,
        log_probs: Float[Array, " batch"],
        task_ids: Float[npt.NDArray, " batch num_tasks"],
    ) -> tuple[Self, LogDict]:
        def alpha_loss(params: FrozenDict) -> Float[Array, ""]:
            log_alpha: jax.Array
            log_alpha = task_ids @ params["params"]["log_alpha"].reshape(-1, 1)  # pyright: ignore [reportAttributeAccessIssue]
            return (-log_alpha * (log_probs + self.target_entropy)).mean()

        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(
            self.alpha.params
        )
        alpha = self.alpha.apply_gradients(grads=alpha_grads)

        return self.replace(alpha=alpha), {
            "losses/alpha_loss": alpha_loss_value,
            "alpha": jnp.exp(alpha.params["params"]["log_alpha"]).sum(),  # pyright: ignore [reportArgumentType]
        }

    @jax.jit
    def _update_inner(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        task_ids = data.observations[..., -self.num_tasks :]

        alpha_vals = self.alpha.apply_fn(self.alpha.params, task_ids)
        if self.use_task_weights:
            task_weights = extract_task_weights(self.alpha.params, task_ids)
        else:
            task_weights = None

        actor_data = critic_data = data
        actor_alpha_vals = critic_alpha_vals = alpha_vals
        actor_task_weights = critic_task_weights = task_weights
        alpha_val_indices = None

        if self.split_critic_losses or self.split_actor_losses:
            split_data, _ = self.split_data_by_tasks(data, task_ids)
            split_alpha_vals, alpha_val_indices = self.split_data_by_tasks(
                alpha_vals, task_ids
            )
            split_task_weights, _ = (
                self.split_data_by_tasks(task_weights, task_ids)
                if task_weights is not None
                else (None, None)
            )

            if self.split_critic_losses:
                critic_data = split_data
                critic_alpha_vals = split_alpha_vals
                critic_task_weights = split_task_weights

            if self.split_actor_losses:
                actor_data = split_data
                actor_alpha_vals = split_alpha_vals
                actor_task_weights = split_task_weights

        self, critic_logs = self.update_critic(
            critic_data, critic_alpha_vals, critic_task_weights
        )
        self, log_probs, actor_logs = self.update_actor(
            actor_data, actor_alpha_vals, actor_task_weights
        )
        if self.split_actor_losses:
            assert alpha_val_indices is not None
            log_probs = self.unsplit_data_by_tasks(log_probs, alpha_val_indices)
        self, alpha_logs = self.update_alpha(log_probs, task_ids)

        # HACK: PCGrad logs
        assert isinstance(self.critic.opt_state, tuple)
        assert isinstance(self.actor.opt_state, tuple)
        critic_optim_logs = (
            {
                f"metrics/critic_{key}": value
                for key, value in self.critic.opt_state[0]._asdict().items()
            }
            if isinstance(self.critic.opt_state[0], PCGradState)
            else {}
        )
        actor_optim_logs = (
            {
                f"metrics/actor_{key}": value
                for key, value in self.actor.opt_state[0]._asdict().items()
            }
            if isinstance(self.actor.opt_state[0], PCGradState)
            else {}
        )

        return self, {
            **critic_logs,
            **actor_logs,
            **alpha_logs,
            **critic_optim_logs,
            **actor_optim_logs,
        }

    @override
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        return self._update_inner(data)
