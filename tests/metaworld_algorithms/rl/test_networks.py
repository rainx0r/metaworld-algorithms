from functools import partial
import chex
import distrax
import jax
import jax.numpy as jnp
import pytest

from metaworld_algorithms.config.networks import ContinuousActionPolicyConfig
from metaworld_algorithms.rl.networks import ContinuousActionPolicy, EnsembleMD


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def policy_and_ensemble():
    policy = partial(ContinuousActionPolicy, action_dim=4, config=ContinuousActionPolicyConfig())
    ensemble_md = EnsembleMD(net_cls=policy, num=3)
    return policy, ensemble_md


def test_expand_params_and_forward_pass(rng, policy_and_ensemble):
    policy, ensemble_md = policy_and_ensemble

    single_params = policy().init(rng, jnp.ones((1, 5)))
    expanded_params = ensemble_md.expand_params(single_params, ensemble_md.num)

    assert "params" in expanded_params
    assert "ensemble" in expanded_params["params"]

    chex.assert_tree_shape_prefix(expanded_params["params"]["ensemble"], (ensemble_md.num,))

    # Generate random data for the ensemble
    data = jax.random.normal(rng, (ensemble_md.num, 10, 5))

    output_ensemble: distrax.Distribution
    output_vmap: distrax.Distribution
    output_ensemble = ensemble_md.apply(expanded_params, data)
    output_vmap = jax.vmap(policy().apply, in_axes=(None, 0), out_axes=0)(single_params, data)
    breakpoint()

    # Check that the outputs are identical
    assert output_ensemble.batch_shape == output_vmap.batch_shape
    assert output_ensemble.event_shape == output_vmap.event_shape
    chex.assert_trees_all_close(output_ensemble.mode(), output_vmap.mode())
    chex.assert_trees_all_close(output_ensemble.stddev(), output_vmap.stddev())
