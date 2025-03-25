from typing import Any, Generator, Never

import scipy
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


def compute_returns(
    rewards: Float[npt.NDArray, "task rollout timestep 1"], discount: float
):
    """Discounted cumulative sum.

    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    """
    # From garage, modified to work on multi-dimensional arrays, and column reward vectors
    reshape = rewards.shape[-1] == 1
    if reshape:
        rewards = rewards.reshape(rewards.shape[:-1])
    returns = scipy.signal.lfilter(
        [1], [1, float(-discount)], rewards[..., ::-1], axis=-1
    )[..., ::-1]
    return returns if not reshape else returns.reshape(*returns.shape, 1)


class LinearFeatureBaseline:
    @staticmethod
    def _extract_features(
        observations: Float[npt.NDArray, "task rollout timestep obs_dim"], reshape=True
    ):
        observations = np.clip(observations, -10, 10)
        ones = np.ones((*observations.shape[:-1], 1))
        timestep = ones * (np.arange(observations.shape[-2]).reshape(-1, 1) / 100.0)
        features = np.concatenate(
            [observations, observations**2, timestep, timestep**2, timestep**3, ones],
            axis=-1,
        )
        if reshape:
            features = features.reshape(features.shape[0], -1, features.shape[-1])
        return features

    @classmethod
    def _fit_baseline(
        cls,
        observations: Float[npt.NDArray, "task rollout timestep obs_dim"],
        returns: Float[npt.NDArray, "task rollout timestep 1"],
        reg_coeff: float = 1e-5,
    ) -> np.ndarray:
        features = cls._extract_features(observations)
        target = returns.reshape(returns.shape[0], -1, 1)

        coeffs = []
        task_coeffs = np.zeros(features.shape[1])
        for task in range(observations.shape[0]):
            featmat = features[task]
            _target = target[task]
            for _ in range(5):
                task_coeffs = np.linalg.lstsq(
                    featmat.T @ featmat + reg_coeff * np.identity(featmat.shape[1]),
                    featmat.T @ _target,
                    rcond=-1,
                )[0]
                if not np.any(np.isnan(task_coeffs)):
                    break
                reg_coeff *= 10

            coeffs.append(np.expand_dims(task_coeffs, axis=0))

        return np.stack(coeffs)

    @classmethod
    def get_baseline_values(
        cls, rollouts: Rollout, discount: float
    ) -> Float[npt.NDArray, "timestep task 1"]:
        assert rollouts.returns is not None

        observations = [[] for _ in range(rollouts.dones.shape[1])]
        rewards = [[] for _ in range(rollouts.dones.shape[1])]
        start_idx = np.zeros(rollouts.dones.shape[1], dtype=np.int32)
        for i in range(rollouts.dones.shape[0]):
            for done in rollouts.dones[i]:
                if done:
                    observations[i].append(rollouts.observations[start_idx[i] : i + 1])
                    rewards[i].append(rollouts.rewards[start_idx[i] : i + 1])
                    start_idx[i] = i + 1

        # NOTE: This will error if the trajectories are not the same length
        observations = np.stack(observations)
        rewards = np.stack(rewards)
        returns = compute_returns(rewards, discount=discount)

        coeffs = cls._fit_baseline(observations, returns)
        features = cls._extract_features(observations, reshape=False)
        return (features @ coeffs).swapaxes(0, 1).reshape(*rollouts.rewards.shape)
