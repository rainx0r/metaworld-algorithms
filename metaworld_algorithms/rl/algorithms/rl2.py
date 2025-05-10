import dataclasses
from collections import defaultdict
from functools import partial
from typing import Self, override

import distrax
import gymnasium as gym
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from flax.linen import FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from metaworld_algorithms.config.envs import MetaLearningEnvConfig
from metaworld_algorithms.config.networks import RecurrentContinuousActionPolicyConfig
from metaworld_algorithms.config.rl import AlgorithmConfig
from metaworld_algorithms.monitoring.utils import (
    Histogram,
    get_logs,
    prefix_dict,
    pytree_histogram,
)
from metaworld_algorithms.nn.distributions import TanhMultivariateNormalDiag
from metaworld_algorithms.rl.algorithms.base import RNNBasedMetaLearningAlgorithm
from metaworld_algorithms.rl.algorithms.utils import (
    LinearFeatureBaseline,
    RNNTrainState,
    compute_gae,
    explained_variance,
    normalize_advantages,
    to_deterministic_minibatch_iterator,
    to_overlapping_chunks,
)
from metaworld_algorithms.rl.networks import RecurrentContinuousActionPolicy
from metaworld_algorithms.types import (
    Action,
    AuxPolicyOutputs,
    LogDict,
    LogProb,
    MetaLearningAgent,
    Observation,
    RNNState,
    Rollout,
    Timestep,
)


@jax.jit
def _sample_action(
    policy: RNNTrainState, state: RNNState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... state_dim"], Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist: distrax.Distribution
    next_state, dist = policy.apply_fn(policy.params, state, observation)
    action = dist.sample(seed=action_key)
    return next_state, action, key


@jax.jit
def _eval_action(
    policy: RNNTrainState, state: RNNState, observation: Observation
) -> tuple[Float[Array, "... state_dim"], Float[Array, "... action_dim"]]:
    dist: distrax.Distribution
    next_state, dist = policy.apply_fn(policy.params, state, observation)
    return next_state, dist.mode()


@jax.jit
def _sample_action_dist(
    policy: RNNTrainState,
    state: RNNState,
    observation: Observation,
    key: PRNGKeyArray,
) -> tuple[
    RNNState,
    Action,
    LogProb,
    Action,
    Action,
    PRNGKeyArray,
]:
    next_state: jax.Array
    dist: distrax.Distribution

    key, action_key = jax.random.split(key)
    next_state, dist = policy.apply_fn(policy.params, state, observation)
    action, action_log_prob = dist.sample_and_log_prob(seed=action_key)

    if isinstance(dist, TanhMultivariateNormalDiag):
        # HACK: use pre-tanh distributions for kl divergence
        mean = dist.pre_tanh_mean()
        std = dist.pre_tanh_std()
    else:
        mean = dist.mode()
        std = dist.stddev()

    return next_state, action, action_log_prob, mean, std, key  # pyright: ignore[reportReturnType]


@dataclasses.dataclass(frozen=True)
class RL2Config(AlgorithmConfig):
    policy_config: RecurrentContinuousActionPolicyConfig = (
        RecurrentContinuousActionPolicyConfig()
    )
    meta_batch_size: int = 20
    clip_eps: float = 0.2
    entropy_coefficient: float = 5e-3
    normalize_advantages: bool = True
    gae_lambda: float = 0.95
    num_epochs: int = 10
    target_kl: float | None = None
    chunk_len: int = 200
    overlap: int = 50


class RL2(RNNBasedMetaLearningAlgorithm[RL2Config]):
    policy: RNNTrainState
    key: PRNGKeyArray
    policy_squash_tanh: bool = struct.field(pytree_node=False)

    gamma: float = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    normalize_advantages: bool = struct.field(pytree_node=False)

    gae_lambda: float = struct.field(pytree_node=False)
    num_epochs: int = struct.field(pytree_node=False)
    target_kl: float | None = struct.field(pytree_node=False)

    chunk_len: int = struct.field(pytree_node=False)
    overlap: int = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: RL2Config,
        env_config: MetaLearningEnvConfig,
        seed: int = 1,
    ) -> "RL2":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert env_config.action_space.shape is not None

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, policy_key = jax.random.split(master_key, 2)

        policy_net = RecurrentContinuousActionPolicy(
            action_dim=int(np.prod(env_config.action_space.shape)),
            config=config.policy_config,
        )

        dummy_obs = jnp.array(
            [
                env_config.observation_space.sample()
                for _ in range(config.meta_batch_size)
            ]
        )
        dummy_carry = policy_net.initialize_carry(config.meta_batch_size, policy_key)

        policy = RNNTrainState.create(
            params=policy_net.init(policy_key, dummy_carry, dummy_obs),
            tx=config.policy_config.network_config.optimizer.spawn(),
            apply_fn=policy_net.apply,
            seq_apply_fn=partial(policy_net.apply, method=policy_net.rollout),
            init_carry_fn=policy_net.initialize_carry,
        )

        return RL2(
            num_tasks=config.num_tasks,
            policy=policy,
            policy_squash_tanh=config.policy_config.squash_tanh,
            key=algorithm_key,
            gamma=config.gamma,
            clip_eps=config.clip_eps,
            entropy_coefficient=config.entropy_coefficient,
            normalize_advantages=config.normalize_advantages,
            gae_lambda=config.gae_lambda,
            num_epochs=config.num_epochs,
            target_kl=config.target_kl,
            chunk_len=config.chunk_len,
            overlap=config.overlap,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "policy_num_params": sum(
                x.size for x in jax.tree.leaves(self.policy.params)
            ),
        }

    def init_recurrent_state(self, batch_size: int) -> tuple[Self, RNNState]:
        key, init_recurrent_key = jax.random.split(self.key)
        carry = self.policy.init_carry_fn(batch_size, init_recurrent_key)
        return self.replace(key=key), carry

    def reset_recurrent_state(
        self, current_state: RNNState, reset_mask: npt.NDArray[np.bool_]
    ) -> tuple[Self, RNNState]:
        self, new_state = self.init_recurrent_state(current_state.shape[0])
        return self, np.where(reset_mask[..., None], new_state, current_state)

    def sample_action_and_aux(
        self, state: RNNState, observation: Observation
    ) -> tuple[Self, RNNState, Action, AuxPolicyOutputs]:
        rets = _sample_action_dist(self.policy, state, observation, self.key)
        state, action, log_prob, mean, std = jax.device_get(rets[:-1])
        key = rets[-1]
        return (
            self.replace(key=key),
            state,
            action,
            {"log_prob": log_prob, "mean": mean, "std": std},
        )

    def sample_action(
        self, state: RNNState, observation: Observation
    ) -> tuple[Self, RNNState, Action]:
        rets = _sample_action(self.policy, state, observation, self.key)
        state, action = jax.device_get(rets[:-1])
        key = rets[-1]
        return (
            self.replace(key=key),
            state,
            action,
        )

    def eval_action(
        self, states: RNNState, observations: Observation
    ) -> tuple[RNNState, Action]:
        return jax.device_get(_eval_action(self.policy, states, observations))

    class RL2Wrapped(MetaLearningAgent):
        _current_state: RNNState
        _adapted_state: RNNState

        def __init__(self, agent: "RL2"):
            self._agent = agent
            self._current_agent = self._agent

        def init(self) -> None:
            self._current_agent, self._current_state = (
                self._current_agent.init_recurrent_state(self._agent.num_tasks)
            )

        def adapt_action(
            self, observations: npt.NDArray[np.float64]
        ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
            self._current_agent, self._current_state, action, aux_policy_outs = (
                self._current_agent.sample_action_and_aux(
                    self._current_state, observations
                )
            )
            return action, aux_policy_outs

        def step(self, timestep: Timestep) -> None:
            pass

        def adapt(self) -> None:
            self._adapted_state = self._current_state.copy()

        def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
            self._current_state = jnp.where(  # pyright: ignore[reportAttributeAccessIssue]
                env_mask[..., None], self._adapted_state, self._current_state
            )

        def eval_action(
            self, observations: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            self._current_state, action = self._current_agent.eval_action(
                self._current_state, observations
            )
            return action

    @override
    def wrap(self) -> MetaLearningAgent:
        return RL2.RL2Wrapped(self)

    def compute_advantages(
        self,
        rollouts: Rollout,
    ) -> Rollout:
        # NOTE: In RL2, we remove episode boundaries in GAE
        # In Rollout, dones is episode_starts in this case
        # We'll just keep the first episode start
        new_dones = np.zeros_like(rollouts.dones)
        new_dones[0] = 1.0
        rollouts = rollouts._replace(dones=new_dones)

        values, returns = LinearFeatureBaseline.get_baseline_values_and_returns(
            rollouts, self.gamma
        )
        rollouts = rollouts._replace(values=values, returns=returns)

        # NOTE: assume the final states are terminal
        dones = np.ones(rollouts.rewards.shape[1:], dtype=rollouts.rewards.dtype)
        rollouts = compute_gae(
            rollouts, self.gamma, self.gae_lambda, last_values=None, dones=dones
        )
        if self.normalize_advantages:
            rollouts = normalize_advantages(rollouts)
        return rollouts

    @jax.jit
    def _update_inner(
        self, data: Rollout, initial_carry: jax.Array
    ) -> tuple[Self, jax.Array, LogDict]:
        def policy_loss(
            params: FrozenDict,
        ) -> tuple[Float[Array, ""], tuple[jax.Array, LogDict]]:
            action_dist: distrax.Distribution
            new_log_probs: Float[Array, " *batch"]
            assert data.log_probs is not None
            assert data.advantages is not None
            assert data.rnn_states is not None
            assert data.valids is not None

            carries, action_dist = self.policy.seq_apply_fn(
                params, data.observations, initial_carry=initial_carry
            )
            new_log_probs = action_dist.log_prob(data.actions)  # pyright: ignore[reportAssignmentType]
            log_ratio = new_log_probs.reshape(data.log_probs.shape) - data.log_probs
            ratio = jnp.exp(log_ratio)

            # For logs
            approx_kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
            clip_fracs = jax.lax.stop_gradient(
                (jnp.abs(ratio - 1.0) > self.clip_eps).mean()
            )

            zero_loss = jnp.zeros_like(data.advantages)

            pg_loss1 = -data.advantages * ratio
            pg_loss2 = -data.advantages * jnp.clip(
                ratio, 1 - self.clip_eps, 1 + self.clip_eps
            )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2)
            pg_loss = jnp.where(data.valids, pg_loss, zero_loss).mean()

            # TODO: Support entropy estimate using log probs
            # also maybe support garage-style entropy term
            entropy_loss = action_dist.entropy()
            entropy_loss = jnp.expand_dims(entropy_loss, -1)
            entropy_loss = jnp.where(data.valids, entropy_loss, zero_loss).mean()

            return pg_loss - self.entropy_coefficient * entropy_loss, (
                carries,
                {
                    "losses/entropy_loss": entropy_loss,
                    "losses/policy_loss": pg_loss,
                    "losses/approx_kl": approx_kl,
                    "losses/clip_fracs": clip_fracs,
                },
            )

        (_, (carries, logs)), policy_grads = jax.value_and_grad(
            policy_loss, has_aux=True
        )(self.policy.params)
        policy_grads_flat, _ = jax.flatten_util.ravel_pytree(policy_grads)
        grads_hist_dict = prefix_dict(
            "nn/policy_grads", pytree_histogram(policy_grads["params"])
        )

        policy = self.policy.apply_gradients(grads=policy_grads)
        policy_params_flat, _ = jax.flatten_util.ravel_pytree(policy.params)
        param_hist_dict = prefix_dict(
            "nn/policy_params", pytree_histogram(policy.params["params"])
        )

        return (
            self.replace(policy=policy),
            carries,
            logs
            | {
                "nn/policy_grad_norm": jnp.linalg.norm(policy_grads_flat),
                "nn/policy_param_norm": jnp.linalg.norm(policy_params_flat),
                **grads_hist_dict,
                **param_hist_dict,
            },
        )

    @jax.jit
    def _get_activations(
        self, data: Rollout
    ) -> tuple[PyTree[Array], PyTree[Array] | None]:
        assert data.rnn_states is not None
        _, policy_state = self.policy.seq_apply_fn(
            self.policy.params,
            data.observations,
            initial_carry=data.rnn_states[0],
            mutable=["intermediates"],
        )
        return policy_state["intermediates"]

    @override
    def update(self, data: Rollout) -> tuple[Self, LogDict]:
        # NOTE: We assume that during training all episodes have the same length
        # This should be the case for Metaworld.
        data = self.compute_advantages(data)  # (rollout_timestep, task, ...)
        data = to_overlapping_chunks(data, self.chunk_len, self.overlap)

        assert data.advantages is not None and data.returns is not None
        assert data.values is not None and data.stds is not None
        assert data.means is not None and data.log_probs is not None
        assert data.rnn_states is not None
        diagnostic_logs = prefix_dict(
            "data",
            {
                **get_logs("advantages", data.advantages),
                **get_logs("returns", data.returns),
                **get_logs("values", data.values),
                **get_logs("rewards", data.rewards),
                **get_logs("rnn_states", data.rnn_states),
                "action_std": Histogram(data.stds.reshape(-1)),
                "action_mean": Histogram(data.means.reshape(-1)),
                "approx_entropy": np.mean(-data.log_probs),
            },
        )

        minibatch_iterator = to_deterministic_minibatch_iterator(data)
        update_logs = defaultdict(list)
        keep_training = True
        for epoch in range(self.num_epochs):
            self, initial_carry = self.init_recurrent_state(data.rewards.shape[2])
            for step in range(len(data.rewards)):
                minibatch_rollout = next(minibatch_iterator)
                self, carries, logs = self._update_inner(
                    minibatch_rollout, initial_carry
                )
                initial_carry = carries[(self.chunk_len - self.overlap - 1)]
                for k, v in logs.items():
                    update_logs[k].append(v)

                if epoch == 0 and step == 0:  # Initial KL and Loss
                    update_logs["metrics/kl_before"] = [logs["losses/approx_kl"]]
                    update_logs["metrics/policy_loss_before"] = [
                        logs["losses/policy_loss"]
                    ]

                if self.target_kl is not None:
                    if logs["losses/approx_kl"] > 1.5 * self.target_kl:
                        print(
                            f"Stopped early at KL {logs['losses/approx_kl']}, ({epoch} epochs)"
                        )
                        keep_training = False
                        break

            if not keep_training:
                break

        # Finalize logs
        final_logs: dict = {
            "metrics/explained_variance": explained_variance(
                data.values.reshape(-1), data.returns.reshape(-1)
            )
        }
        for k, v in update_logs.items():
            if not isinstance(v[0], Histogram):
                final_logs[k] = np.mean(v)
            else:
                # TODO: should probably not be just the last histogram
                final_logs[k] = v[-1]

        # log activations
        policy_acts = self._get_activations(next(minibatch_iterator))
        final_logs.update(prefix_dict("nn/activations", pytree_histogram(policy_acts)))

        return self, diagnostic_logs | final_logs
