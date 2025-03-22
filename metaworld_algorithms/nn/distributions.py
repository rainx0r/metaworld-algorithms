import distrax

from typing import override

import jax
import jax.numpy as jnp
import chex


class TanhMultivariateNormalDiag(distrax.Transformed):
    """Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/distributions.py#L13"""

    def __init__(self, loc: jax.Array, scale_diag: jax.Array) -> None:
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        super().__init__(
            distribution=distribution, bijector=distrax.Block(distrax.Tanh(), 1)
        )

    @override
    def log_prob(self, value: chex.Array) -> chex.Array:
        # HACK: The value is undefined at -1.0 and 1.0, yet sometimes such a value can arise
        # from numerical stability issues with < float64 precision.
        # Simple fix is to just bound the action to just under -1.0 and 1.0
        value = jnp.clip(value, -1.0 + 1e-7, 1.0 - 1e-7)
        return super().log_prob(value)

    @override
    def entropy(self, input_hint: chex.Array | None = None) -> chex.Array:
        # TODO: This is most likely mathematically inaccurate, can we do better?
        return self.distribution.entropy()  # pyright: ignore [reportReturnType]

    @override
    def kl_divergence(self, other_dist, **kwargs) -> chex.Array:
        if isinstance(other_dist, TanhMultivariateNormalDiag):
            # TODO: use pre-tanh distributions for kl divergence
            # not entirely sure if this is mathematically accurate
            return self.distribution.kl_divergence(other_dist.distribution, **kwargs)
        else:
            return super().kl_divergence(other_dist, **kwargs)

    def pre_tanh_mean(self) -> jax.Array:
        return self.distribution.loc  # pyright: ignore [reportReturnType]

    def pre_tanh_std(self) -> jax.Array:
        return self.distribution.scale_diag  # pyright: ignore [reportReturnType]

    @override
    def stddev(self) -> jax.Array:
        return self.bijector.forward(self.distribution.stddev())  # pyright: ignore [reportReturnType]

    @override
    def mode(self) -> jax.Array:
        return self.bijector.forward(self.distribution.mode())  # pyright: ignore [reportReturnType]
