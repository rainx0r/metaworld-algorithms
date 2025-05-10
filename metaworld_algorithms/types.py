from typing import Any, NamedTuple, Protocol, TypedDict

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float
from metaworld_algorithms.monitoring.utils import Histogram

Action = Float[np.ndarray, "... action_dim"]
Value = Float[np.ndarray, "... 1"]
LogProb = Float[np.ndarray, "... 1"]
Observation = Float[np.ndarray, "... obs_dim"]
RNNState = Float[np.ndarray, "... rnn_state_dim"]
LayerActivations = Float[Array, "batch_size layer_dim"]

type LogDict = dict[str, float | Float[Array, ""] | Histogram]
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
    rnn_states: Float[RNNState, "timestep task"] | None = None

    # Computed statistics about observed rewards
    returns: Float[np.ndarray, "timestep task 1"] | None = None
    advantages: Float[np.ndarray, "timestep task 1"] | None = None
    valids: Float[np.ndarray, "episode timestep 1"] | None = None

    @classmethod
    def from_list(cls, items: list) -> "Rollout":
        if Timestep.is_timestep(items[0]):
            items = list(map(lambda x: Timestep.to_rollout(x), items))
        return cls(*map(lambda *xs: np.stack(xs), *items))


class Timestep(NamedTuple):
    observation: npt.NDArray
    action: npt.NDArray
    reward: npt.NDArray
    terminated: npt.NDArray
    truncated: npt.NDArray
    aux_policy_outputs: dict[str, npt.NDArray]

    @classmethod
    def is_timestep(cls, item: Any) -> bool:
        return hasattr(item, "_fields") and item._fields == cls._fields

    @staticmethod
    def to_rollout(item: "Timestep") -> Rollout:
        log_probs = item.aux_policy_outputs.get("log_prob")
        if log_probs is not None:
            log_probs = log_probs[..., None]

        return Rollout(
            observations=item.observation,
            actions=item.action,
            rewards=item.reward[..., None],
            dones=item.truncated[..., None],
            log_probs=log_probs,
            means=item.aux_policy_outputs.get("mean"),
            stds=item.aux_policy_outputs.get("std"),
            values=item.aux_policy_outputs.get("value"),
        )


class Agent(Protocol):
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None: ...


class MetaLearningAgent(Agent, Protocol):
    def init(self) -> None: ...

    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]: ...

    def step(self, timestep: Timestep) -> None: ...

    def adapt(self) -> None: ...


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
