import numpy as np


def test_shuffle_is_deterministic():
    # Set a fixed seed for reproducibility
    seed = 42

    # Create two arrays of the same shape
    arr1 = np.arange(10)
    arr2 = np.arange(10)
    arr3 = np.arange(10)

    generator = np.random.default_rng(seed)
    original_state = generator.bit_generator.state
    generator.shuffle(arr1)

    generator.bit_generator.state = original_state
    generator.shuffle(arr2)
    assert np.array_equal(arr1, arr2)

    generator.shuffle(arr3)
    assert not np.array_equal(arr1, arr3)
