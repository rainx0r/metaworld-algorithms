import dataclasses
from typing import Self, override

import jax
from flax import struct
from jaxtyping import PRNGKeyArray

from metaworld_algorithms.config.envs import MetaLearningEnvConfig
from metaworld_algorithms.config.rl import AlgorithmConfig
from metaworld_algorithms.rl.algorithms.base import RNNBasedMetaLearningAlgorithm
from metaworld_algorithms.rl.algorithms.utils import MetaTrainState
from metaworld_algorithms.types import (
    Action,
    AuxPolicyOutputs,
    LogDict,
    Observation,
    RNNState,
    Rollout,
)


@dataclasses.dataclass(frozen=True)
class RL2Config(AlgorithmConfig):
    policy_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    meta_batch_size: int = 20
    clip_eps: float = 0.2
    entropy_coefficient: float = 5e-3
    normalize_advantages: bool = True
    gae_lambda: float = 0.97
    num_gradient_steps: int = 32
    num_epochs: int = 16
    target_kl: float | None = None


class RL2(RNNBasedMetaLearningAlgorithm[RL2Config]):
    policy: MetaTrainState
    key: PRNGKeyArray
    policy_squash_tanh: bool = struct.field(pytree_node=False)

    gamma: float = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    normalize_advantages: bool = struct.field(pytree_node=False)

    gae_lambda: float = struct.field(pytree_node=False)
    num_gradient_steps: int = struct.field(pytree_node=False)
    num_epochs: int = struct.field(pytree_node=False)
    target_kl: float | None = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: RL2Config,
        env_config: MetaLearningEnvConfig,
        seed: int = 1,
    ) -> "RL2":
        # TODO:

        return RL2(
            num_tasks=config.num_tasks,
            policy=policy,
            policy_squash_tanh=config.policy_config.squash_tanh,
            key=algorithm_key,
            gamma=config.gamma,
            clip_eps=config.clip_eps,
            entropy_coefficient=config.entropy_coefficient,
            normalize_advantages=config.normalize_advantages,
            gae_lambda=config.gae_lambda,
            num_gradient_steps=config.num_gradient_steps,
            num_epochs=config.num_epochs,
            target_kl=config.target_kl,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        # TODO: network param counts
        return {}

    def init_recurrent_state(self, batch_size: int) -> tuple[Self, RNNState]:
        # TODO:
        return self, ...

    def sample_action_and_aux(
        self, state: RNNState, observation: Observation
    ) -> tuple[Self, RNNState, Action, AuxPolicyOutputs]:
        # TODO: action sampling
        return ...

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        # TODO: The interface is wrong here, maybe move sample_action to be inside
        # non-ML algorithm subclasses?
        return ...

    @override
    def eval_action(self, observations: Observation) -> Action:
        # TODO: eval action
        return ...

    @override
    def wrap(self) -> MetaLearningAgent:
        # TODO: RL2 metalearning agent
        return ...

    @jax.jit
    def _update_inner(self, data: Rollout) -> tuple[Self, LogDict]:
        # TODO: Recurrent PPO update
        return self, {}

    def update(self, data: Rollout) -> tuple[Self, LogDict]:
        # TODO: minibatching methods
        return ...
