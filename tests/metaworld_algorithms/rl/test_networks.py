from functools import partial

import chex
import jax
import jax.numpy as jnp
import pytest
from flax.core import FrozenDict

from metaworld_algorithms.config.networks import ContinuousActionPolicyConfig
from metaworld_algorithms.nn.base import MLP
from metaworld_algorithms.nn.distributions import TanhMultivariateNormalDiag
from metaworld_algorithms.rl.networks import (
    ContinuousActionPolicy,
    EnsembleMD,
    EnsembleMDContinuousActionPolicy,
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


def test_expand_params_and_forward_pass(rng):
    net = partial(MLP, head_dim=4, width=10, depth=3)
    ensemble_md = EnsembleMD(net_cls=net, num=3)

    single_params = net().init(rng, jnp.ones((10, 5)))
    expanded_params = ensemble_md.expand_params(single_params)

    assert "params" in expanded_params
    assert "ensemble" in expanded_params["params"]

    chex.assert_tree_shape_prefix(
        expanded_params["params"]["ensemble"], (ensemble_md.num,)
    )

    # Generate random data for the ensemble
    data = jax.random.normal(rng, (ensemble_md.num, 10, 5))

    output_ensemble = ensemble_md.apply(expanded_params, data)
    output_vmap = jax.vmap(net().apply, in_axes=(None, 0), out_axes=0)(
        single_params, data
    )

    assert jnp.allclose(output_ensemble, output_vmap)  # pyright: ignore[reportArgumentType]


def test_expand_params_and_forward_pass_continuous_action_policy(rng):
    policy = ContinuousActionPolicy(action_dim=4, config=ContinuousActionPolicyConfig())
    ensemble_md = EnsembleMDContinuousActionPolicy(
        action_dim=4, num=3, config=ContinuousActionPolicyConfig()
    )

    single_params = ensemble_md.init_single(rng, jnp.ones((10, 5)))
    expanded_params = ensemble_md.expand_params(single_params)

    assert "params" in expanded_params
    assert "ensemble" in expanded_params["params"]

    chex.assert_tree_shape_prefix(
        expanded_params["params"]["ensemble"], (ensemble_md.num,)
    )

    # Generate random data for the ensemble
    data = jax.random.normal(rng, (ensemble_md.num, 10, 5))

    output_ensemble: TanhMultivariateNormalDiag
    output_ensemble = ensemble_md.apply(expanded_params, data)  # pyright: ignore [reportAssignmentType]
    policy_params = FrozenDict(
        {"params": {"ContinuousActionPolicyTorso_0": single_params["params"]}}
    )
    single_outs: list[TanhMultivariateNormalDiag] = [
        policy.apply(policy_params, data[i]) for i in range(ensemble_md.num)
    ]  # pyright: ignore [reportAssignmentType]
    modes = jnp.stack([out.mode() for out in single_outs])
    stds = jnp.stack([out.distribution.scale_diag for out in single_outs])

    assert output_ensemble.batch_shape == (3, 10)
    assert output_ensemble.event_shape == (4,)

    # Check that the outputs are identical
    assert jnp.allclose(output_ensemble.mode(), modes)
    assert jnp.allclose(output_ensemble.distribution.scale_diag, stds)
