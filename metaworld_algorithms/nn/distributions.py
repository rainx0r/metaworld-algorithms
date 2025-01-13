import distrax

import jax


class TanhMultivariateNormalDiag(distrax.Transformed):
    """From https://github.com/kevinzakka/nanorl/blob/main/nanorl/distributions.py#L13"""

    def __init__(self, loc: jax.Array, scale_diag: jax.Array) -> None:
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        super().__init__(
            distribution=distribution, bijector=distrax.Block(distrax.Tanh(), 1)
        )

    def mode(self) -> jax.Array:
        return self.bijector.forward(self.distribution.mode())  # pyright: ignore [reportReturnType]
