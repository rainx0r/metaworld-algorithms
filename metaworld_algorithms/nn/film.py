import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from metaworld_algorithms.config.nn import FiLMConfig

from .base import MLP


class FiLMNetwork(nn.Module):
    config: FiLMConfig

    head_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]
        x = x[..., : -self.config.num_tasks]

        # Task Encoder
        # TODO: This should be replaced with a pretrained NLP model eventually
        # for language description embeddings
        task_embedding = nn.Embed(
            num_embeddings=self.config.num_tasks,
            features=self.config.embedding_dim,
            name="task_idx_embedding",
        )(task_idx.argmax(-1))
        chex.assert_shape(
            task_embedding, (*task_idx.shape[:-1], self.config.embedding_dim)
        )
        task_embedding = self.config.activation(task_embedding)

        film_gammas_and_betas = MLP(
            width=self.config.encoder_width,
            depth=self.config.encoder_depth,
            head_dim=2 * (self.config.encoder_depth + 1),
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
            activation_fn=self.config.activation,
        )(task_embedding)
        # (batch_dims, num_encoder_layers + 1, 2, 1)
        # gamma and beta for each encoder layer, scalar input to avoid weird broadcasting
        film_gammas_and_betas = film_gammas_and_betas.reshape(
            *film_gammas_and_betas.shape[:-1], (self.config.encoder_depth + 1), 2, 1
        )
        chex.assert_shape(
            film_gammas_and_betas,
            (*task_idx.shape[:-1], (self.config.encoder_depth + 1), 2, 1),
        )

        # FiLM Obs encoder
        for i in range(self.config.encoder_depth):
            x = nn.Dense(
                self.config.encoder_width,
                kernel_init=self.config.kernel_init(),
                bias_init=self.config.bias_init(),
                use_bias=self.config.use_bias,
            )(x)
            x = self.config.activation(x)
            x = (
                x * film_gammas_and_betas[..., i, 0, :]
                + film_gammas_and_betas[..., i, 1, :]
            )
            self.sow("intermediates", f"encoder_layer_{i}", x)
        x = nn.Dense(
            self.config.embedding_dim,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
        )(x)
        encoder_out = (
            x * film_gammas_and_betas[..., -1, 0, :]
            + film_gammas_and_betas[..., -1, 1, :]
        )

        torso_input = jnp.concatenate((encoder_out, task_embedding), axis=-1)
        self.sow("intermediates", "encoder_output", torso_input)
        chex.assert_shape(
            torso_input,
            (*x.shape[:-1], self.config.embedding_dim + self.config.embedding_dim),
        )

        torso_output = MLP(
            head_dim=self.head_dim,
            depth=self.config.depth,
            width=self.config.width,
            activation_fn=self.config.activation,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
            head_kernel_init=self.head_kernel_init,
            head_bias_init=self.head_bias_init,
            name="torso",
        )(torso_input)
        chex.assert_shape(torso_output, (*x.shape[:-1], self.head_dim))

        return torso_output
