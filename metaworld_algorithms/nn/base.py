from collections.abc import Callable

import flax.linen as nn
import jax

from .utils import name_prefix


class MLP(nn.Module):
    """A Flax Module to represent an MLP feature extractor."""

    head_dim: int

    depth: int = 3
    # TODO: Support variable width?
    width: int = 400

    activation_fn: Callable[[jax.typing.ArrayLike], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.constant(0.1)
    use_bias: bool = True

    head_kernel_init: jax.nn.initializers.Initializer | None = None
    head_bias_init: jax.nn.initializers.Initializer | None = None
    activate_last: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i in range(self.depth):
            x = nn.Dense(
                self.width,
                name=f"layer_{i}",
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.use_bias,
            )(x)
            x = self.activation_fn(x)
            self.sow("intermediates", f"{name_prefix(self)}layer_{i}", x)
        x = nn.Dense(
            self.head_dim,
            name=f"layer_{self.depth}",
            kernel_init=self.head_kernel_init or self.kernel_init,
            bias_init=self.head_bias_init or self.bias_init,
            use_bias=self.use_bias,
        )(x)
        if self.activate_last:
            x = self.activation_fn(x)
        return x
