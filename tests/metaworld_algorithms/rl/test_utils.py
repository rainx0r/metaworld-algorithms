import numpy as np
from metaworld_algorithms.rl.algorithms.utils import (
    LinearFeatureBaseline,
    compute_gae,
    normalize_advantages,
    to_minibatch_iterator,
)
from metaworld_algorithms.types import Rollout
import itertools


def test_minibatch_iterator():
    # Create the Rollout object
    batch_size = 10
    observations = np.array(
        [np.full(39, i) for i in range(batch_size)]
    )  # Shape (10, 39)
    actions = np.array([np.full(4, i) for i in range(batch_size)])  # Shape (10, 4)
    rewards = np.array([[i] for i in range(batch_size)])  # Shape (10, 1)
    dones = np.array([[i] for i in range(batch_size)])  # Shape (10, 1)
    data = Rollout(observations, actions, rewards, dones)

    # Test parameters
    num_minibatches = 2
    seed = 42
    num_epochs = 5

    # Create the iterator
    iterator = to_minibatch_iterator(data, num_minibatches, seed)
    previous_minibatch_rewards = None

    # Run the test
    for _ in range(num_epochs):
        for _ in range(num_minibatches):
            minibatch = next(iterator)
            # Check alignment
            for j in range(minibatch.observations.shape[0]):
                assert np.all(minibatch.observations[j] == minibatch.observations[j][0])
                assert np.all(minibatch.actions[j] == minibatch.actions[j][0])
                assert (
                    minibatch.observations[j][0]
                    == minibatch.actions[j][0]
                    == minibatch.rewards[j][0]
                    == minibatch.dones[j][0]
                )

            # Check shuffling via rewards
            rewards_flat = minibatch.rewards.flatten()
            if previous_minibatch_rewards is not None:
                assert not np.array_equal(rewards_flat, previous_minibatch_rewards)
            previous_minibatch_rewards = rewards_flat
            assert not np.array_equal(
                rewards_flat, np.arange(batch_size // num_minibatches)
            )


def test_linear_feature_baseline(metarl_rollouts: Rollout):
    assert metarl_rollouts.returns is not None
    assert metarl_rollouts.advantages is not None
    assert metarl_rollouts.values is not None

    values, returns = LinearFeatureBaseline.get_baseline_values_and_returns(
        metarl_rollouts, 0.99
    )
    assert np.allclose(returns, metarl_rollouts.returns)
    assert np.allclose(values, metarl_rollouts.values)

    rollouts = metarl_rollouts._replace(values=values, returns=returns)
    final_dones = np.ones((1, *rollouts.dones.shape[1:]))
    rollouts_with_advantages = compute_gae(rollouts, 0.99, 0.97, None, final_dones)
    rollouts_with_normalised_advantages = normalize_advantages(rollouts_with_advantages)
    assert rollouts_with_normalised_advantages.advantages is not None
    assert np.allclose(
        rollouts_with_normalised_advantages.advantages, metarl_rollouts.advantages
    )
