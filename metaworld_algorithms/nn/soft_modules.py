import flax.linen as nn
import jax
import jax.numpy as jnp

from metaworld_algorithms.config.nn import SoftModulesConfig

from .base import MLP
from .initializers import uniform
from .utils import name_prefix

# NOTE: the paper is missing quite a lot of details that are in the official code
#
# 1) there is an extra embedding layer for the task embedding after z and f have been combined
#    that downsizes the embedding from D to 256 (in both the deep and shallow versions of the network)
# 2) the obs embedding is activated before it's passed into the layers
# 3) p_l+1 is not dependent on just p_l but on all p_<l with skip connections
# 4) ReLU is applied after the weighted sum in forward computation, not before as in Eq. 8 in the paper
# 5) there is an extra p_L+1 that is applied as a dot product over the final module outputs
# 6) the task weights take the softmax over log alpha, not actual alpha.
#    And they're also multiplied by the number of tasks
#
# These are marked with "NOTE: <number>"


class BasePolicyNetworkLayer(nn.Module):
    """A Flax Module to represent a single layer of modules of the Base Policy Network"""

    config: SoftModulesConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        modules = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},  # Different params per module
            split_rngs={"params": True},  # Different initialization per module
            in_axes=-2,
            out_axes=-2,
            axis_size=self.config.num_modules,
        )(
            self.config.module_width,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
        )

        # NOTE: 4, activation *should* be here according to the paper, but it's after the weighted sum
        return modules(x)


class RoutingNetworkLayer(nn.Module):
    """A Flax Module to represent a single layer of the Routing Network"""

    config: SoftModulesConfig
    last: bool = False  # NOTE: 5

    def setup(self):
        self.prob_embedding_fc = nn.Dense(
            self.config.width,  # TODO: double check but this does not seem to be D
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
        )  # W_u^l
        # NOTE: 5
        self.prob_output_dim = (
            self.config.num_modules
            if self.last
            else self.config.num_modules * self.config.num_modules
        )
        self.prob_output_fc = nn.Dense(
            self.prob_output_dim,
            kernel_init=uniform(1e-3),
            bias_init=jax.nn.initializers.zeros,
            use_bias=self.config.use_bias,
        )  # W_d^l

    def __call__(
        self, task_embedding: jax.Array, prev_probs: jax.Array | None = None
    ) -> jax.Array:
        if prev_probs is not None:  # Eq 5-only bit
            task_embedding *= self.prob_embedding_fc(prev_probs)
            self.sow(
                "intermediates", f"{name_prefix(self)}task_embedding", task_embedding
            )
        x = self.prob_output_fc(self.config.activation(task_embedding))
        if not self.last:  # NOTE: 5
            x = x.reshape(
                *x.shape[:-1], self.config.num_modules, self.config.num_modules
            )
        x = jax.nn.softmax(x, axis=-1)  # Eq. 7
        return x


class SoftModularizationNetwork(nn.Module):
    """A Flax Module to represent the Base Policy Network and the Routing Network simultaneously,
    since their layers are so intertwined.

    Corresponds to `ModularGatedCascadeCondNet` in the official implementation."""

    config: SoftModulesConfig

    head_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    routing_skip_connections: bool = True  # NOTE: 3

    def setup(self) -> None:
        # Base policy network layers
        self.f = MLP(
            depth=1,
            head_dim=self.config.embedding_dim,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
            activation_fn=self.config.activation,
        )
        self.layers = [
            BasePolicyNetworkLayer(self.config) for _ in range(self.config.depth)
        ]
        self.output_head = nn.Dense(
            self.head_dim,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
            use_bias=self.config.use_bias,
        )

        # Routing network layers
        self.z = MLP(
            depth=0,
            head_dim=self.config.embedding_dim,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
        )
        self.task_embedding_fc = MLP(
            depth=1,
            width=self.config.width,
            head_dim=self.config.width,
            activation_fn=self.config.activation,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
        )  # NOTE: 1
        self.prob_fcs = [
            RoutingNetworkLayer(self.config, last=i == self.config.depth - 1)
            for i in range(self.config.depth)  # NOTE: 5
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]
        x = x[..., : -self.config.num_tasks]

        # Feature extraction
        obs_embedding = self.f(x)
        task_embedding = self.z(task_idx) * obs_embedding
        # NOTE: 1
        task_embedding = self.task_embedding_fc(self.config.activation(task_embedding))

        # Initial layer inputs
        prev_probs = None
        obs_embedding = self.config.activation(obs_embedding)  # NOTE: 2
        self.sow("intermediates", "task_embedding", task_embedding)
        self.sow("intermediates", "obs_embedding", obs_embedding)
        module_ins = jnp.stack(
            [obs_embedding for _ in range(self.config.num_modules)], axis=-2
        )
        weights = None

        if self.routing_skip_connections:  # NOTE: 3
            weights = []

        # Equation 8, holds for all layers except L
        for i in range(self.config.depth - 1):
            probs = self.prob_fcs[i](task_embedding, prev_probs)
            self.sow("intermediates", f"layer_{i}_probs", probs.reshape(x.shape[0], -1))
            # NOTE: 4
            module_outs = self.config.activation(probs @ self.layers[i](module_ins))
            self.sow(
                "intermediates",
                f"layer_{i}_modules",
                module_outs.reshape(x.shape[0], -1),
            )

            # Post processing
            probs = probs.reshape(
                *probs.shape[:-2], self.config.num_modules * self.config.num_modules
            )
            if weights is not None and self.routing_skip_connections:  # NOTE: 3
                weights.append(probs)
                prev_probs = jnp.concatenate(weights, axis=-1)
            else:
                prev_probs = probs
            module_ins = module_outs

        # Last layer L, Equation 9
        module_outs = self.layers[-1](module_ins)
        probs = jnp.expand_dims(
            self.prob_fcs[-1](task_embedding, prev_probs), axis=-1
        )  # NOTE: 5
        self.sow(
            "intermediates",
            f"layer_{self.config.depth - 1}_probs",
            probs.reshape(x.shape[0], -1),
        )
        output_embedding = self.config.activation(jnp.sum(module_outs * probs, axis=-2))
        self.sow(
            "intermediates",
            f"layer_{self.config.depth - 1}_modules",
            output_embedding.reshape(x.shape[0], -1),
        )
        return self.output_head(output_embedding)
