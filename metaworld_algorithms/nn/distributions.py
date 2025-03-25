import distrax

from typing import Sequence, override

from distrax._src.distributions.distribution import EventT, IntLike, PRNGKey
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

    def _clip(self, sample: chex.Array) -> chex.Array:
        # NOTE: Normally, a Tanh-transformed distribution is bounded in (-1,1).
        # but due to numerical stability issues, it's possible that some samples are
        # exactly 1.0 or -1.0. Thus, we can clip samples to be always valid, using
        # the smallest representable float as the epsilon, given by `finfo` for the
        # current data type.
        clip_bound = 1.0 - jnp.finfo(sample.dtype).eps
        return jnp.clip(sample, -clip_bound, clip_bound)

    @override
    def sample(
        self, *, seed: IntLike | PRNGKey, sample_shape: IntLike | Sequence[IntLike] = ()
    ) -> chex.Array:
        sample = super().sample(seed=seed, sample_shape=sample_shape)
        return self._clip(sample)

    @override
    def sample_and_log_prob(
        self, *, seed: IntLike | PRNGKey, sample_shape: IntLike | Sequence[IntLike] = ()
    ) -> tuple[chex.Array, chex.Array]:
        sample, log_prob = super().sample_and_log_prob(
            seed=seed, sample_shape=sample_shape
        )
        return self._clip(sample), log_prob

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
