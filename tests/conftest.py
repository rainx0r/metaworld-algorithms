import pytest
import numpy as np
from pathlib import Path
from metaworld_algorithms.rl.buffers import Rollout

DATA_PATH = Path(__file__).parent / "data"


@pytest.fixture
def metarl_rollouts():
    ROLLOUTS_PATH = DATA_PATH / "rollouts_0"
    with open(ROLLOUTS_PATH / "observations.npy", "rb") as f:
        observations = np.load(f).swapaxes(0, 1)
    with open(ROLLOUTS_PATH / "rewards.npy", "rb") as f:
        rewards = np.load(f).swapaxes(0, 1)
    with open(ROLLOUTS_PATH / "returns.npy", "rb") as f:
        returns = np.load(f).swapaxes(0, 1)
    with open(ROLLOUTS_PATH / "advantages.npy", "rb") as f:
        advantages = np.load(f).swapaxes(0, 1)
    with open(ROLLOUTS_PATH / "dones.npy", "rb") as f:
        dones = np.load(f).swapaxes(0, 1)
    with open(ROLLOUTS_PATH / "values.npy", "rb") as f:
        values = np.load(f)

    actions = np.ones((*observations.shape[:-1], 4), dtype=np.float64)
    return Rollout(
        observations,
        actions,
        rewards,
        dones,
        values=values,
        returns=returns,
        advantages=advantages,
    )
