import optax
import jax


def dummy_multitask_optimizer() -> optax.GradientTransformation:
    def init(params: optax.Params) -> dict:
        del params
        return {}

    @jax.jit
    def update(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, dict]:
        del state
        del params
        return jax.tree.map(lambda x: x.mean(axis=0), updates), {}

    return optax.GradientTransformation(init=init, update=update)
