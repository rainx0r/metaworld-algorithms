import numpy as np
from metaworld_algorithms.rl.algorithms.utils import to_minibatch_iterator
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
    for epoch in range(num_epochs):
        for minibatch in itertools.islice(iterator, num_minibatches):
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
                assert not np.array_equal(
                    rewards_flat, previous_minibatch_rewards
                )
            previous_minibatch_rewards = rewards_flat
            assert not np.array_equal(
                rewards_flat, np.arange(batch_size // num_minibatches)
            )
