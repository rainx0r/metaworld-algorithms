from collections.abc import Callable
from functools import cached_property, partial

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
    ValueFunctionConfig,
)
from metaworld_algorithms.config.utils import StdType
from metaworld_algorithms.nn import get_nn_arch_for_config
from metaworld_algorithms.nn.distributions import TanhMultivariateNormalDiag
from metaworld_algorithms.nn.initializers import uniform


class ContinuousActionPolicyTorso(nn.Module):
    """A Flax module representing the torso of the policy network for continous action spaces."""

    action_dim: int
    config: ContinuousActionPolicyConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        mlp_head_dim = self.action_dim
        if self.config.std_type == StdType.MLP_HEAD:
            mlp_head_dim *= 2

        head_kernel_init = uniform(1e-3)
        if self.config.head_kernel_init is not None:
            head_kernel_init = self.config.head_kernel_init()

        head_bias_init = uniform(1e-3)
        if self.config.head_bias_init is not None:
            head_bias_init = self.config.head_bias_init()

        x = get_nn_arch_for_config(self.config.network_config)(
            config=self.config.network_config,
            head_dim=mlp_head_dim,
            head_kernel_init=head_kernel_init,
            head_bias_init=head_bias_init,
        )(x)

        if self.config.std_type == StdType.MLP_HEAD:
            mean, log_std = jnp.split(x, 2, axis=-1)
        elif self.config.std_type == StdType.PARAM:
            mean = x
            log_std = self.param(  # init std to 1
                "log_std", nn.initializers.zeros_init(), (self.action_dim,)
            )
            log_std = jnp.broadcast_to(log_std, mean.shape)
        else:
            raise ValueError("Invalid std_type: %s" % self.config.std_type)

        log_std = jnp.clip(
            log_std, a_min=self.config.log_std_min, a_max=self.config.log_std_max
        )
        std = jnp.exp(log_std)

        return mean, std


class ContinuousActionPolicy(nn.Module):
    """A Flax module representing the policy network for continous action spaces."""

    action_dim: int
    config: ContinuousActionPolicyConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> distrax.Distribution:
        mean, std = ContinuousActionPolicyTorso(
            action_dim=self.action_dim, config=self.config
        )(x)

        if self.config.squash_tanh:
            return TanhMultivariateNormalDiag(loc=mean, scale_diag=std)
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)


class QValueFunction(nn.Module):
    """A Flax module approximating a Q-Value function."""

    config: QValueFunctionConfig

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        # NOTE: certain NN architectures that make use of task IDs will be looking for them
        # at the last N_TASKS dimensions of their input. So while normally concat(state,action) makes more sense
        # we'll go with (action, state) here
        x = jnp.concatenate((action, state), axis=-1)

        if not self.config.use_classification:
            return get_nn_arch_for_config(self.config.network_config)(
                config=self.config.network_config,
                head_dim=1,
                head_kernel_init=uniform(3e-3),
                head_bias_init=uniform(3e-3),
            )(x)
        else:
            raise NotImplementedError(
                "Value prediction as classification is not supported yet."
            )


class ValueFunction(nn.Module):
    """A Flax module approximating a Q-Value function."""

    config: ValueFunctionConfig

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        if not self.config.use_classification:
            return get_nn_arch_for_config(self.config.network_config)(
                config=self.config.network_config,
                head_dim=1,
                head_kernel_init=uniform(3e-3),
                head_bias_init=uniform(3e-3),
            )(state)
        else:
            raise NotImplementedError(
                "Value prediction as classification is not supported yet."
            )


class Ensemble(nn.Module):
    net_cls: nn.Module | Callable[..., nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


class EnsembleMDContinuousActionPolicy(nn.Module):
    """Ensemble ContinusActionPolicy where there is "multiple data" as input.
    That is, the in_axes in the vmap is not None, and axis 0 should correspond
    to the ensemble num."""

    # HACK: We need this rather than a truly generic EnsembleMD class cause of a bug
    # distrax when using vmap and MultivariateNormalDiag
    # - https://github.com/google-deepmind/distrax/issues/239
    # - https://github.com/google-deepmind/distrax/issues/276
    # Can probably just fix the bug and contribute upstream but this will do for now

    action_dim: int
    num: int
    config: ContinuousActionPolicyConfig

    @cached_property
    def _net_cls(self) -> Callable[..., nn.Module]:
        return partial(
            ContinuousActionPolicyTorso,
            action_dim=self.action_dim,
            config=self.config,
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> distrax.Distribution:
        ensemble = nn.vmap(
            self._net_cls,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=0,
            out_axes=0,
            axis_size=self.num,
        )
        mean, std = ensemble(name="ensemble")(x)

        if self.config.squash_tanh:
            return TanhMultivariateNormalDiag(loc=mean, scale_diag=std)
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

    def init_single(self, rng: PRNGKeyArray, x: jax.Array) -> nn.FrozenDict | dict:
        return self._net_cls(parent=None).init(rng, x)

    def expand_params(self, params: nn.FrozenDict | dict) -> nn.FrozenDict:
        inner_params = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.num,) + x.shape), params
        )["params"]
        return nn.FrozenDict({"params": {"ensemble": inner_params}})


class EnsembleMD(nn.Module):
    """Ensemble where there is "multiple data" as input.
    That is, the in_axes in the vmap is not None, and axis 0 should correspond
    to the ensemble num."""

    net_cls: nn.Module | Callable[..., nn.Module]
    num: int

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=0,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble(name="ensemble")(*args)

    def expand_params(self, params: nn.FrozenDict | dict) -> nn.FrozenDict:
        inner_params = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.num,) + x.shape), params
        )["params"]
        return nn.FrozenDict({"params": {"ensemble": inner_params}})
