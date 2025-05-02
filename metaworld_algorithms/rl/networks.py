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
    RecurrentContinuousActionPolicyConfig,
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
            log_std, min=self.config.log_std_min, max=self.config.log_std_max
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


class RecurrentContinuousActionPolicy(nn.Module):
    """A Flax module representing the policy network for continous action spaces."""

    action_dim: int
    config: RecurrentContinuousActionPolicyConfig

    def _get_cell(self, **kwargs) -> nn.RNNCellBase:
        return self.config.network_config.cell_type(
            features=self.config.network_config.width,
            kernel_init=self.config.network_config.kernel_init(),
            recurrent_kernel_init=self.config.network_config.recurrent_kernel_init(),
            bias_init=self.config.network_config.bias_init(),
            **kwargs,
        )

    def setup(self) -> None:
        # TODO: abstract this into a nn module?
        # Might be useful so we can support transformers instead etc

        self.encoder = None
        if self.config.encoder_config is not None:
            self.encoder = get_nn_arch_for_config(self.config.encoder_config)(
                config=self.config.encoder_config,
                head_dim=self.config.encoder_config.width,
                head_kernel_init=self.config.encoder_config.kernel_init(),
                head_bias_init=self.config.encoder_config.bias_init(),
                activate_last=True,
            )

        self.cell: nn.RNNCellBase = self._get_cell()

        def scan_fn(cell: nn.RNNCellBase, carry: jax.Array, x: jax.Array):
            carry, y = cell(carry, x)
            return carry, (carry, y)

        self.rnn = nn.scan(
            scan_fn,
            in_axes=0,
            out_axes=0,
            unroll=1,
            variable_broadcast="params",
            split_rngs={"params": False},
        )

        head_kernel_init = uniform(1e-3)
        if self.config.head_kernel_init is not None:
            head_kernel_init = self.config.head_kernel_init()

        head_bias_init = uniform(1e-3)
        if self.config.head_bias_init is not None:
            head_bias_init = self.config.head_bias_init()

        _Dense = partial(
            nn.Dense,
            kernel_init=head_kernel_init,
            bias_init=head_bias_init,
            use_bias=self.config.network_config.use_bias,
        )
        if self.config.std_type == StdType.MLP_HEAD:
            self.head = _Dense(self.action_dim * 2)
        elif self.config.std_type == StdType.PARAM:
            self.head = _Dense(self.action_dim)
            self.log_std = self.param(  # init std to 1
                "log_std", nn.initializers.zeros_init(), (self.action_dim,)
            )
        else:
            raise ValueError("Invalid std_type: %s" % self.config.std_type)

    def _process_head(self, x: jax.Array) -> distrax.Distribution:
        if self.config.std_type == StdType.MLP_HEAD:
            mean, log_std = jnp.split(x, 2, axis=-1)
        elif self.config.std_type == StdType.PARAM:
            mean = x
            log_std = jnp.broadcast_to(self.log_std, mean.shape)
        else:
            raise ValueError("Invalid std_type: %s" % self.config.std_type)

        log_std = jnp.clip(
            log_std, min=self.config.log_std_min, max=self.config.log_std_max
        )
        std = jnp.exp(log_std)

        if self.config.squash_tanh:
            return TanhMultivariateNormalDiag(loc=mean, scale_diag=std)
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

    def initialize_carry(self, batch_size: int, key: PRNGKeyArray) -> jax.Array:
        # NOTE: Second dim doesn't matter in initialize_carry
        return self._get_cell(parent=None).initialize_carry(key, (batch_size, 1))

    def __call__(
        self, carry: jax.Array, x: jax.Array
    ) -> tuple[jax.Array, distrax.Distribution]:
        if self.encoder is not None:
            x = self.encoder(x)
            self.sow("intermediates", "encoder_out", x)
        carry, x = self.cell(carry, x)
        self.sow("intermediates", "rnn_cell", x)
        out = self._process_head(self.head(x))
        return carry, out

    def rollout(
        self,
        x: jax.Array,
        initial_carry: jax.Array,
    ) -> tuple[jax.Array, distrax.Distribution]:
        if self.encoder is not None:
            x = self.encoder(x)
            self.sow("intermediates", "encoder_out", x)

        _, (carries, x) = self.rnn(self.cell, initial_carry, x)
        self.sow("intermediates", "rnn_carries", carries)
        self.sow("intermediates", "rnn", x)
        out = self._process_head(self.head(x))
        return carries, out


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
