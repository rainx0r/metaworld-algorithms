from typing import Any, Generator, Never

import numpy as np
import numpy.typing as npt
import optax
from flax import struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training.train_state import TrainState as FlaxTrainState
from jaxtyping import Float
from typing_extensions import Callable

from metaworld_algorithms.types import Rollout


class TrainState(FlaxTrainState):
    def apply_gradients(
        self,
        *,
        grads,
        optimizer_extra_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads["params"]
            params_with_opt = self.params["params"]
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        if optimizer_extra_args is None:
            optimizer_extra_args = {}

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt, **optimizer_extra_args
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                "params": new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


class MetaTrainState(TrainState):
    inner_train_state: TrainState
    expand_params: Callable = struct.field(pytree_node=False)


def to_minibatch_iterator(
    data: Rollout, num: int, seed: int
) -> Generator[Rollout, None, Never]:
    # Flatten batch dims
    rollouts = Rollout(
        *map(
            lambda x: x.reshape(-1, x.shape[-1]) if x is not None else None,
            data,
        )  # pyright: ignore[reportArgumentType]
    )

    rollout_size = rollouts.observations.shape[0]
    minibatch_size = rollout_size // num

    rng = np.random.default_rng(seed)
    rng_state = rng.bit_generator.state

    while True:
        for field in data:
            rng.bit_generator.state = rng_state
            if field is not None:
                rng.shuffle(field)
        rng_state = rng.bit_generator.state
        for start in range(0, rollout_size, minibatch_size):
            end = start + minibatch_size
            yield Rollout(
                *map(
                    lambda x: x[start:end] if x is not None else None,  # pyright: ignore[reportArgumentType]
                    data,
                )
            )


def compute_gae(
    rollouts: Rollout,
    gamma: float,
    gae_lambda: float,
    last_values: Float[npt.NDArray, " task"] | None,
    dones: Float[npt.NDArray, " task"],
) -> Rollout:
    assert rollouts.values is not None

    if last_values is not None:
        last_values = last_values.reshape(-1, 1)
    else:
        if np.all(dones == 1.0):
            last_values = np.zeros_like(rollouts.values)
        else:
            raise ValueError(
                "Must provide final value estimates if the final timestep is not terminal for all envs."
            )
    dones = dones.reshape(-1, 1)

    advantages = np.zeros_like(rollouts.rewards)

    # Adapted from https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py
    last_gae_lamda = 0
    num_rollout_steps = rollouts.observations.shape[0]
    for timestep in reversed(range(num_rollout_steps)):
        if timestep == num_rollout_steps - 1:
            next_nonterminal = 1.0 - dones
            next_values = last_values
        else:
            next_nonterminal = 1.0 - rollouts.dones[timestep + 1]
            next_values = rollouts.values[timestep + 1]
        delta = (
            rollouts.rewards[timestep]
            + next_nonterminal * gamma * next_values
            - rollouts.values[timestep]
        )
        advantages[timestep] = last_gae_lamda = (
            delta + next_nonterminal * gamma * gae_lambda * last_gae_lamda
        )

    returns = advantages + rollouts.values

    if not hasattr(rollouts, "returns"):
        # NOTE: Can't use `replace` here if this is a Rollout from MetaWorld's evaluation interface
        return Rollout(
            returns=returns,
            advantages=advantages,
            observations=rollouts.observations,
            actions=rollouts.actions,
            rewards=rollouts.rewards,
            dones=rollouts.dones,
            log_probs=rollouts.log_probs,
            means=rollouts.means,
            stds=rollouts.stds,
            values=rollouts.values,
        )
    else:
        return rollouts._replace(
            returns=returns,
            advantages=advantages,
        )
