import dataclasses
from typing import Self, override

import distrax
import gymnasium as gym
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from flax import struct
from flax.core import FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray

from metaworld_algorithms.config.envs import MetaLearningEnvConfig
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    ValueFunctionConfig,
)
from metaworld_algorithms.config.rl import AlgorithmConfig
from metaworld_algorithms.nn.distributions import TanhMultivariateNormalDiag
from metaworld_algorithms.rl.algorithms.utils import MetaTrainState, TrainState
from metaworld_algorithms.rl.networks import (
    EnsembleMDContinuousActionPolicy,
)
from metaworld_algorithms.types import (
    Action,
    AuxPolicyOutputs,
    LogDict,
    LogProb,
    MetaLearningAgent,
    Observation,
    Rollout,
)

from .base import GradientBasedMetaLearningAlgorithm
from .utils import (
    LinearFeatureBaseline,
    compute_gae,
    dones_to_episode_starts,
    normalize_advantages,
    swap_rollout_axes,
)


@jax.jit
def _sample_action(
    policy: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action(
    policy: TrainState, observation: Observation
) -> Float[Array, "... action_dim"]:
    dist: distrax.Distribution
    dist = policy.apply_fn(policy.params, observation)
    return dist.mode()


@jax.jit
def _sample_action_dist(
    policy: TrainState,
    observation: Observation,
    key: PRNGKeyArray,
) -> tuple[
    Action,
    LogProb,
    Action,
    Action,
    PRNGKeyArray,
]:
    key, action_key = jax.random.split(key)
    dist = policy.apply_fn(policy.params, observation)
    action, action_log_prob = dist.sample_and_log_prob(seed=action_key)

    if isinstance(dist, TanhMultivariateNormalDiag):
        # HACK: use pre-tanh distributions for kl divergence
        mean = dist.pre_tanh_mean()
        std = dist.pre_tanh_std()
    else:
        mean = dist.mode()
        std = dist.stddev()

    return action, action_log_prob, mean, std, key  # pyright: ignore[reportReturnType]


@dataclasses.dataclass(frozen=True)
class MAMLTRPOConfig(AlgorithmConfig):
    policy_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    policy_inner_lr: float = 0.1
    vf_config: ValueFunctionConfig = ValueFunctionConfig()
    meta_batch_size: int = 20
    delta: float = 0.01
    cg_iters: int = 10
    backtrack_ratio: float = 0.8
    max_backtrack_iters: int = 15
    gae_lambda: float = 0.97


class MAMLTRPO(GradientBasedMetaLearningAlgorithm[MAMLTRPOConfig]):
    policy: MetaTrainState
    key: PRNGKeyArray
    policy_squash_tanh: bool = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)
    delta: float = struct.field(pytree_node=False)
    cg_iters: int = struct.field(pytree_node=False)
    backtrack_ratio: float = struct.field(pytree_node=False)
    max_backtrack_iters: int = struct.field(pytree_node=False)
    policy_inner_lr: float = struct.field(pytree_node=False)
    gae_lambda: float = struct.field(pytree_node=False)

    @override
    def init_ensemble_networks(self) -> Self:
        policy = self.policy.replace(
            inner_train_state=self.policy.inner_train_state.replace(
                params=self.policy.expand_params(self.policy.params)
            )
        )
        return self.replace(policy=policy)

    @override
    @staticmethod
    def initialize(
        config: MAMLTRPOConfig,
        env_config: MetaLearningEnvConfig,
        seed: int = 1,
    ) -> "MAMLTRPO":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)

        algorithm_key, policy_key = jax.random.split(master_key, 2)
        policy_net = EnsembleMDContinuousActionPolicy(
            num=config.meta_batch_size,
            action_dim=int(np.prod(env_config.action_space.shape)),
            config=config.policy_config,
        )

        dummy_obs = jnp.array(
            [
                env_config.observation_space.sample()
                for _ in range(config.meta_batch_size)
            ]
        )
        policy = MetaTrainState.create(
            params=policy_net.init_single(policy_key, dummy_obs),
            tx=config.policy_config.network_config.optimizer.spawn(),
            inner_train_state=TrainState.create(
                params=dict(),
                tx=optax.sgd(learning_rate=config.policy_inner_lr),
                apply_fn=policy_net.apply,
            ),
            expand_params=policy_net.expand_params,
            apply_fn=None,
        )

        return MAMLTRPO(
            num_tasks=config.num_tasks,
            gamma=config.gamma,
            delta=config.delta,
            cg_iters=config.cg_iters,
            backtrack_ratio=config.backtrack_ratio,
            max_backtrack_iters=config.max_backtrack_iters,
            policy=policy,
            policy_squash_tanh=config.policy_config.squash_tanh,
            key=algorithm_key,
            policy_inner_lr=config.policy_inner_lr,
            gae_lambda=config.gae_lambda,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "policy_num_params": sum(
                x.size for x in jax.tree.leaves(self.policy.params)
            ),
        }

    @override
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]:
        rets = _sample_action_dist(self.policy.inner_train_state, observation, self.key)
        action, log_prob, mean, std = jax.device_get(rets[:-1])
        key = jax.device_get(rets[-1])
        return (
            self.replace(key=key),
            action,
            {"log_prob": log_prob, "mean": mean, "std": std},
        )

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(
            self.policy.inner_train_state, observation, self.key
        )
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observations: Observation) -> Action:
        return jax.device_get(_eval_action(self.policy.inner_train_state, observations))

    @override
    def adapt(self, rollouts: Rollout) -> Self:
        rollouts = self.compute_advantages(rollouts)
        rollouts = swap_rollout_axes(rollouts, 0, 1)
        policy = self.policy.replace(
            inner_train_state=self.inner_step(self.policy.inner_train_state, rollouts)
        )
        return self.replace(policy=policy)

    class MAMLTRPOWrapped(MetaLearningAgent):
        def __init__(self, agent: "MAMLTRPO"):
            self.agent = agent.init_ensemble_networks()

        def adapt_action(
            self, observations: npt.NDArray[np.float64]
        ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
            self.agent, action, aux_policy_outs = self.agent.sample_action_and_aux(
                observations
            )
            return action, aux_policy_outs

        def adapt(self, rollouts: Rollout) -> None:
            # MetaWorld's evaluation stores done instead of episode_start
            # TODO: Maybe just change the interface?
            rollouts = dones_to_episode_starts(rollouts)
            rollouts = self.agent.compute_advantages(rollouts)
            self.agent = self.agent.adapt(rollouts)

        def eval_action(
            self, observations: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            return self.agent.eval_action(observations)

    @override
    def wrap(self) -> MetaLearningAgent:
        return MAMLTRPO.MAMLTRPOWrapped(self)

    def compute_advantages(
        self,
        rollouts: Rollout,
    ) -> Rollout:
        # NOTE: assume the final states are terminal
        dones = np.full(rollouts.rewards.shape[1:], 1.0, dtype=rollouts.rewards.dtype)
        values, returns = LinearFeatureBaseline.get_baseline_values_and_returns(
            rollouts, self.gamma
        )
        if not hasattr(rollouts, "returns"):
            # For Rollout from MetaWorld's evaluation interface
            # TODO: Maybe just change the interface?
            rollouts = Rollout(*rollouts)
        rollouts = rollouts._replace(values=values, returns=returns)
        rollouts = compute_gae(
            rollouts, self.gamma, self.gae_lambda, last_values=None, dones=dones
        )
        rollouts = normalize_advantages(rollouts)
        return rollouts

    @jax.jit
    def inner_step(self, policy: TrainState, rollouts: Rollout) -> TrainState:
        def inner_opt_objective(_theta: FrozenDict):
            log_probs = jnp.expand_dims(
                policy.apply_fn(_theta, rollouts.observations).log_prob(
                    rollouts.actions
                ),
                -1,
            )
            return -(log_probs * rollouts.advantages).mean()

        grads = jax.grad(inner_opt_objective)(policy.params)
        updated_policy = policy.apply_gradients(grads=grads)  # Inner gradient step

        return updated_policy

    @jax.jit
    def outer_step(
        self,
        all_rollouts: list[Rollout],
    ) -> tuple[Self, LogDict]:
        def maml_loss(theta: FrozenDict):
            vec_theta = self.policy.expand_params(theta)
            inner_train_state = self.policy.inner_train_state.replace(params=vec_theta)

            # Adaptation steps
            for i in range(len(all_rollouts) - 1):
                rollouts = all_rollouts[i]
                inner_train_state = self.inner_step(inner_train_state, rollouts)

            # Inner Train State now has theta^\prime
            # Compute MAML objective
            rollouts = all_rollouts[-1]
            new_param_dist = inner_train_state.apply_fn(
                inner_train_state.params, rollouts.observations
            )
            new_param_log_probs = jnp.expand_dims(
                new_param_dist.log_prob(rollouts.actions), -1
            )

            likelihood_ratio = jnp.exp(new_param_log_probs - rollouts.log_probs)
            outer_objective = likelihood_ratio * rollouts.advantages
            return -outer_objective.mean()

        # TRPO, outer gradient step
        def kl_constraint(
            params: FrozenDict, inputs: list[Rollout], targets: distrax.Distribution
        ):
            vec_theta = self.policy.expand_params(params)
            inner_train_state = self.policy.inner_train_state.replace(params=vec_theta)

            # Adaptation steps
            for i in range(len(inputs) - 1):
                rollouts = inputs[i]
                inner_train_state = self.inner_step(inner_train_state, rollouts)

            new_param_dist = inner_train_state.apply_fn(
                inner_train_state.params, inputs[-1].observations
            )
            return targets.kl_divergence(new_param_dist).mean()

        target_dist_cls = (
            TanhMultivariateNormalDiag
            if self.policy_squash_tanh
            else distrax.MultivariateNormalDiag
        )
        target_dist = target_dist_cls(
            loc=all_rollouts[-1].means,  # pyright: ignore[reportArgumentType]
            scale_diag=all_rollouts[-1].stds,  # pyright: ignore[reportArgumentType]
        )
        kl_before = kl_constraint(self.policy.params, all_rollouts, target_dist)

        ## Compute search direction by solving for Ax = g

        def hvp(x):
            hvp_deep = optax.second_order.hvp(
                kl_constraint,  # pyright: ignore[reportArgumentType]
                v=x,
                params=self.policy.params,
                inputs=all_rollouts,  # pyright: ignore[reportArgumentType]
                targets=target_dist,  # pyright: ignore[reportArgumentType]
            )
            hvp_shallow = jax.flatten_util.ravel_pytree(hvp_deep)[0]
            return hvp_shallow + 1e-5 * x  # Ensure positive definite

        loss_before, opt_objective_grads = jax.value_and_grad(maml_loss)(
            self.policy.params
        )
        g, unravel_params = jax.flatten_util.ravel_pytree(opt_objective_grads)
        s, _ = jax.scipy.sparse.linalg.cg(hvp, g, maxiter=self.cg_iters)

        ## Compute optimal step beta
        beta = jnp.sqrt(2.0 * self.delta * (1 / (jnp.dot(s, hvp(s)) + 1e-7)))

        ## Line search
        s = unravel_params(s)

        def _cond_fn(val):
            step, loss, kl, _ = val
            return ((kl > self.delta) | (loss >= loss_before)) & (
                step < self.max_backtrack_iters
            )

        def _body_fn(val):
            step, loss, kl, _ = val
            new_params = jax.tree_util.tree_map(
                lambda theta_i, s_i: theta_i
                - (self.backtrack_ratio**step) * beta * s_i,
                self.policy.params,
                s,
            )
            loss, kl = (
                maml_loss(new_params),
                kl_constraint(new_params, all_rollouts, target_dist),
            )
            return step + 1, loss, kl, new_params

        step, loss, kl, new_params = jax.lax.while_loop(
            _cond_fn,
            _body_fn,
            init_val=(0, loss_before, jnp.finfo(jnp.float32).max, self.policy.params),
        )

        # Param updates
        # Reject params if line search failed
        params = jax.lax.cond(
            (loss < loss_before) & (kl <= self.delta),
            lambda: new_params,
            lambda: self.policy.params,
        )
        policy = self.policy.replace(params=params)

        return self.replace(policy=policy), {
            "losses/loss_before": jnp.mean(loss_before),
            "losses/loss_after": jnp.mean(loss),
            "losses/kl_before": kl_before,
            "losses/kl_after": jnp.array(kl),
            "losses/backtrack_steps": step,
        }

    @override
    def update(self: Self, data: list[Rollout]) -> tuple[Self, LogDict]:
        data = [self.compute_advantages(rollouts) for rollouts in data]
        # Update policy (MetaRL outer step)
        data = [swap_rollout_axes(rollouts, 0, 1) for rollouts in data]
        self, policy_logs = self.outer_step(data)
        return self, policy_logs
