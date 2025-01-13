import jax


def uniform(bound: float) -> jax.nn.initializers.Initializer:
    def _init(key: jax.Array, shape, dtype):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init  # pyright: ignore [reportReturnType]
