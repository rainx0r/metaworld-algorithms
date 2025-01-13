import flax.linen as nn
import jax.numpy as jnp


class L2Normalize(nn.Module):
    @nn.compact
    def __call__(self, x):
        norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True))
        return x / (norm + 1e-8)
