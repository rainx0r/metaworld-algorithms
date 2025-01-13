import flax.linen as nn
import jax
import jax.numpy as jnp

from metaworld_algorithms.config.nn import MultiHeadConfig
from metaworld_algorithms.nn.regularizers import L2Normalize


class MultiHeadNetwork(nn.Module):
    config: MultiHeadConfig

    head_dim: int
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros
    normalize_layer: bool = False
    skip_connection: bool = False

    # TODO: support variable width?

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        batch_dim = x.shape[0]
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]
        skip = None
        for i in range(self.config.depth):
            if self.normalize_layer and i == 0:
                x = L2Normalize()(x)
            if self.skip_connection and x.shape[1] == self.config.width:
                skip = x
            x = nn.Dense(
                self.config.width,
                name=f"layer_{i}",
                kernel_init=self.config.kernel_init(),
                bias_init=self.config.bias_init(),
                use_bias=self.config.use_bias,
            )(x)
            if self.skip_connection and skip is not None:
                x = x + skip
                skip = None
            x = self.config.activation(x)
            self.sow("intermediates", f"torso_layer_{i}", x)

        # 2) Create a head for each task. Pass *every* input through *every* head
        # because we assume the batch dim is not necessarily a task dimension

        x = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=1,
            axis_size=self.config.num_tasks,
        )(
            self.head_dim,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
            use_bias=self.config.use_bias,
        )(x)

        # 3) Collect the output from the appropriate head for each input
        task_indices = task_idx.argmax(axis=-1)
        x = x[jnp.arange(batch_dim), task_indices]

        return x
