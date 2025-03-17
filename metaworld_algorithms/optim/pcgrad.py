from functools import partial
from typing import Any, NamedTuple

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float


class PCGradState(NamedTuple):
    n_grad_conflicts: Float[Array, ""]
    avg_grad_magnitude: Float[Array, ""]
    avg_grad_magnitude_before_surgery: Float[Array, ""]
    avg_cosine_similarity: Float[Array, ""]
    avg_cosine_similarity_diff: Float[Array, ""]


def pcgrad(
    num_tasks: int, cosine_sim_logs: bool = False
) -> optax.GradientTransformation:
    def pcgrad_init(params: optax.Params) -> PCGradState:
        del params
        base_state = PCGradState(
            n_grad_conflicts=jnp.array(0),
            avg_grad_magnitude=jnp.array(0.0),
            avg_grad_magnitude_before_surgery=jnp.array(0.0),
            avg_cosine_similarity=jnp.array(0.0)
            if cosine_sim_logs
            else jnp.array(jnp.nan),
            avg_cosine_similarity_diff=jnp.array(0.0)
            if cosine_sim_logs
            else jnp.array(jnp.nan),
        )
        return base_state

    @jax.jit
    def pcgrad_update(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, PCGradState]:
        del state
        chex.assert_tree_shape_prefix(updates, (num_tasks,))
        assert params is not None

        key: jax.Array | None
        key = extra_args.get("key")
        assert key is not None, "RNG key must be provided"

        def _pcgrad(
            task_grads: Float[Array, "task num_params"],
        ) -> tuple[Float[Array, "task num_params"], Float[Array, ""]]:
            @partial(jax.vmap, in_axes=(0, None), out_axes=0)
            def p_grads(
                task_gradient: Float[Array, " num_params"],
                all_grads: Float[Array, " num_tasks num_params"],
            ) -> tuple[Float[Array, " num_params"], Float[Array, ""]]:
                g_i_pc = task_gradient
                total = 0
                for j in range(all_grads.shape[0]):
                    proj_direction = jnp.dot(g_i_pc, all_grads[j]) / (
                        (all_grads[j] ** 2).sum() + 1e-8
                    )
                    g_i_pc = g_i_pc - jnp.minimum(proj_direction, 0.0) * all_grads[j]
                    confs = (proj_direction < 0).sum()
                    total += confs
                return g_i_pc, jnp.array(total)

            # (num_tasks, gradient_dim)
            res, total = p_grads(task_grads, task_grads)
            total = total.sum() / 2
            return res, total

        flat_task_gradients = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(
            updates
        )
        flat_task_gradients = jax.random.permutation(key, flat_task_gradients)
        final_grads, total_grad_conflicts = _pcgrad(flat_task_gradients)
        avg_grad = final_grads.mean(axis=0)

        new_state = PCGradState(
            n_grad_conflicts=total_grad_conflicts,
            avg_grad_magnitude=(jnp.linalg.norm(final_grads, axis=1)).mean(),
            avg_grad_magnitude_before_surgery=(
                jnp.linalg.norm(flat_task_gradients, axis=1)
            ).mean(),
            avg_cosine_similarity=jnp.array(jnp.nan),
            avg_cosine_similarity_diff=jnp.array(jnp.nan),
        )

        # Compute additional cosine similarity metrics if requested
        if cosine_sim_logs:

            def vmap_cos_sim(grads, num_tasks):
                def calc_cos_sim(selected_grad, grads):
                    new_cos_sim = jnp.array(
                        [
                            jnp.sum(selected_grad * grads, axis=1)
                            / (
                                jnp.linalg.norm(selected_grad)
                                * jnp.linalg.norm(grads, axis=1)
                                + 1e-8
                            )
                        ]
                    )
                    return new_cos_sim

                cos_sim_mat = jax.vmap(calc_cos_sim, in_axes=(0, None), out_axes=-1)(
                    grads, grads
                )
                # Get upper triangle
                mask = jnp.triu(jnp.ones((num_tasks, num_tasks)), k=1)
                num_unique = mask.sum()

                masked_cos_sim = mask * cos_sim_mat
                # n in upper triangle
                avg_cos_sim = masked_cos_sim.flatten().sum() / (num_unique + 1e-8)
                return avg_cos_sim

            avg_cos_sim = vmap_cos_sim(flat_task_gradients, num_tasks)
            new_cos_sim = vmap_cos_sim(final_grads, num_tasks)
            new_state = new_state._replace(
                avg_cosine_similarity=avg_cos_sim,
                avg_cosine_similarity_diff=avg_cos_sim - new_cos_sim,
            )

        _, unravel_fn = jax.flatten_util.ravel_pytree(params)
        return unravel_fn(avg_grad), new_state

    return optax.GradientTransformationExtraArgs(
        init=pcgrad_init,
        update=pcgrad_update,
    )
