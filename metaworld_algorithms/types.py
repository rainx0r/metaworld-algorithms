from typing import Any, NamedTuple, Protocol, TypedDict

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float

Action = Float[np.ndarray, "... action_dim"]
Value = Float[np.ndarray, "... 1"]
LogProb = Float[np.ndarray, "... 1"]
Observation = Float[np.ndarray, "... obs_dim"]
LayerActivations = Float[Array, "batch_size layer_dim"]

type LogDict = dict[str, float | Float[Array, ""]]
type AuxPolicyOutputs = dict[str, npt.NDArray]
type LayerActivationsDict = dict[str, Float[Array, "batch_size layer_dim"]]
type Intermediates = dict[str, tuple[LayerActivations, ...] | "Intermediates"]


class ReplayBufferSamples(NamedTuple):
    observations: Float[Observation, " batch"]
    actions: Float[Action, " batch"]
    next_observations: Float[Observation, " batch"]
    dones: Float[np.ndarray, "batch 1"]
    rewards: Float[np.ndarray, "batch 1"]


class Rollout(NamedTuple):
    # Standard timestep data
    observations: Float[Observation, "timestep task"]
    actions: Float[Action, "timestep task"]
    rewards: Float[np.ndarray, "timestep task 1"]
    dones: Float[np.ndarray, "timestep task 1"]

    # Auxiliary policy outputs
    log_probs: Float[LogProb, "timestep task"] | None = None
    means: Float[Action, "timestep task"] | None = None
    stds: Float[Action, "timestep task"] | None = None
    values: Float[np.ndarray, "timestep task 1"] | None = None

    # Computed statistics about observed rewards
    returns: Float[np.ndarray, "timestep task 1"] | None = None
    advantages: Float[np.ndarray, "timestep task 1"] | None = None


class Agent(Protocol):
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...


class MetaLearningAgent(Agent, Protocol):
    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]: ...

    def adapt(self, rollouts: Rollout) -> None: ...


class CheckpointMetadata(TypedDict):
    timestamp: str
    step: int
    episodes_ended: int


class RNGCheckpoint(TypedDict):
    python_rng_state: tuple[Any, ...]
    global_numpy_rng_state: dict[str, Any]


class ReplayBufferCheckpoint(TypedDict):
    data: dict[str, npt.NDArray[np.float32] | int | bool]
    rng_state: Any


type GymVectorEnv = gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv
type EnvCheckpoint = list[tuple[str, dict[str, Any]]]
