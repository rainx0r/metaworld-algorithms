import dataclasses
import itertools
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
from jaxtyping import Array, Float, PRNGKeyArray

from metaworld_algorithms.config.envs import MetaLearningEnvConfig
from metaworld_algorithms.config.networks import RecurrentContinuousActionPolicyConfig
from metaworld_algorithms.config.rl import AlgorithmConfig
from metaworld_algorithms.nn.distributions import TanhMultivariateNormalDiag
from metaworld_algorithms.rl.algorithms.base import RNNBasedMetaLearningAlgorithm
from metaworld_algorithms.rl.algorithms.utils import (
    LinearFeatureBaseline,
    RNNTrainState,
    compute_gae,
    normalize_advantages,
    to_episode_batch,
    to_minibatch_iterator,
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
    gae_lambda: float = 0.97
    num_gradient_steps: int = 20
    num_epochs: int = 16
    target_kl: float | None = None


class RL2(RNNBasedMetaLearningAlgorithm[RL2Config]):
    policy: RNNTrainState
    key: PRNGKeyArray
    policy_squash_tanh: bool = struct.field(pytree_node=False)

    gamma: float = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    normalize_advantages: bool = struct.field(pytree_node=False)

    gae_lambda: float = struct.field(pytree_node=False)
    num_gradient_steps: int = struct.field(pytree_node=False)
    num_epochs: int = struct.field(pytree_node=False)
    target_kl: float | None = struct.field(pytree_node=False)

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
            num_gradient_steps=config.num_gradient_steps,
            num_epochs=config.num_epochs,
            target_kl=config.target_kl,
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

        def init(self) -> None:
            self._current_agent = self._agent
            self._current_agent, self._current_state = self._agent.init_recurrent_state(
                self._agent.num_tasks
            )

        def adapt_action(
            self, observations: npt.NDArray[np.float64]
        ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
            self._current_agent, self._current_state, action, aux_policy_outs = (
                self._agent.sample_action_and_aux(self._current_state, observations)
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
        # NOTE: assume the final states are terminal
        dones = np.ones(rollouts.rewards.shape[1:], dtype=rollouts.rewards.dtype)
        values, returns = LinearFeatureBaseline.get_baseline_values_and_returns(
            rollouts, self.gamma
        )

        # NOTE: In RL2, we remove episode boundaries in GAE
        # In Rollout, dones is episode_starts in this case
        # We'll just keep the first episode start
        new_dones = np.zeros_like(rollouts.dones)
        new_dones[0] = 1.0
        rollouts = rollouts._replace(values=values, returns=returns, dones=new_dones)

        rollouts = compute_gae(
            rollouts, self.gamma, self.gae_lambda, last_values=None, dones=dones
        )
        if self.normalize_advantages:
            rollouts = normalize_advantages(rollouts)
        return rollouts

    @jax.jit
    def _update_inner(self, data: Rollout) -> tuple[Self, LogDict]:
        def policy_loss(params: FrozenDict) -> tuple[Float[Array, ""], LogDict]:
            action_dist: distrax.Distribution
            new_log_probs: Float[Array, " *batch"]
            assert data.log_probs is not None
            assert data.advantages is not None
            assert data.rnn_states is not None

            action_dist = self.policy.seq_apply_fn(
                params, data.observations, initial_carry=data.rnn_states[0]
            )
            new_log_probs = action_dist.log_prob(data.actions)  # pyright: ignore[reportAssignmentType]
            log_ratio = new_log_probs.reshape(data.log_probs.shape) - data.log_probs
            ratio = jnp.exp(log_ratio)

            # For logs
            approx_kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
            clip_fracs = jax.lax.stop_gradient(
                (jnp.abs(ratio - 1.0) > self.clip_eps).mean()
            )

            pg_loss1 = -data.advantages * ratio
            pg_loss2 = -data.advantages * jnp.clip(
                ratio, 1 - self.clip_eps, 1 + self.clip_eps
            )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # TODO: Support entropy estimate using log probs
            entropy_loss = action_dist.entropy().mean()

            return pg_loss - self.entropy_coefficient * entropy_loss, {
                "losses/entropy_loss": entropy_loss,
                "losses/policy_loss": pg_loss,
                "losses/approx_kl": approx_kl,
                "losses/clip_fracs": clip_fracs,
            }

        (_, logs), policy_grads = jax.value_and_grad(policy_loss, has_aux=True)(
            self.policy.params
        )
        policy_grads_flat, _ = jax.flatten_util.ravel_pytree(policy_grads)
        policy = self.policy.apply_gradients(grads=policy_grads)
        policy_params_flat, _ = jax.flatten_util.ravel_pytree(policy.params)

        return self.replace(policy=policy), logs | {
            "metrics/policy_grad_magnitude": jnp.linalg.norm(policy_grads_flat),
            "metrics/policy_param_norm": jnp.linalg.norm(policy_params_flat),
        }

    @override
    def update(self, data: Rollout) -> tuple[Self, LogDict]:
        # NOTE: We assume that during training all episodes have the same length
        # This should be the case for Metaworld.
        data = self.compute_advantages(data)  # (rollout_timestep, task, ...)
        # TODO: get the horizon from somewhere?
        # or better yet switch to properly padded data with masking
        data = to_episode_batch(data, 500)  # (episode, ep_timestep, ...)

        # NOTE: Minibatch over rollouts
        # Pick random rollouts from the data for each minibatch, but use the whole episode
        key, minibatch_iterator_key = jax.random.split(self.key)
        self = self.replace(key=key)
        seed = jax.random.randint(
            minibatch_iterator_key, (), minval=0, maxval=jnp.iinfo(jnp.int32).max
        ).item()
        minibatch_iterator = to_minibatch_iterator(
            data, self.num_gradient_steps, int(seed)
        )

        logs = {}
        for epoch in range(self.num_epochs):
            for minibatch_rollout in itertools.islice(
                minibatch_iterator, self.num_gradient_steps
            ):
                self, logs = self._update_inner(minibatch_rollout)

            if self.target_kl is not None:
                if logs["losses/approx_kl"] > self.target_kl:
                    print(
                        f"Stopped early at KL {logs['losses/approx_kl']}, ({epoch} epochs)"
                    )
                    break

        return self, logs
