import flax.linen as nn
import jax
import jax.numpy as jnp

from metaworld_algorithms.config.nn import PaCoConfig


def CompositionalDense(num_parameter_sets: int):
    return nn.vmap(
        nn.Dense,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,  # pyright: ignore [reportArgumentType]
        out_axes=-2,
        axis_size=num_parameter_sets,
    )


class PaCoNetwork(nn.Module):
    config: PaCoConfig

    head_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]
        x = x[..., : -self.config.num_tasks]

        # Task ID embedding
        w_tau = nn.Dense(
            self.config.num_parameter_sets,
            use_bias=self.config.use_bias,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
        )(task_idx)

        for i in range(self.config.depth):
            x = CompositionalDense(self.config.num_parameter_sets)(
                self.config.width,
                use_bias=self.config.use_bias,
                kernel_init=self.config.kernel_init(),
                bias_init=self.config.bias_init(),
            )(x)
            x = jnp.einsum("bkn,bk->bn", x, w_tau)
            x = self.config.activation(x)
            self.sow("intermediates", f"paco_layer_{i}", x)

        x = CompositionalDense(self.config.num_parameter_sets)(
            self.head_dim,
            use_bias=self.config.use_bias,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
        )(x)
        x = jnp.einsum("bkn,bk->bn", x, w_tau)

        return x
