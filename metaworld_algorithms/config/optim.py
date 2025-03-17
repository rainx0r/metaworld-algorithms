from dataclasses import dataclass

import jax
import optax

from metaworld_algorithms.optim.dummy import dummy_multitask_optimizer
from metaworld_algorithms.optim.gradnorm import gradnorm
from metaworld_algorithms.optim.pcgrad import pcgrad

from .utils import Optimizer


@dataclass(frozen=True, kw_only=True)
class OptimizerConfig:
    lr: float = 3e-4
    optimizer: Optimizer = Optimizer.Adam
    max_grad_norm: float | None = None

    @property
    def requires_split_task_losses(self) -> bool:
        return False

    def spawn(self) -> optax.GradientTransformation:
        # From https://github.com/araffin/sbx/blob/master/sbx/ppo/policies.py#L120
        optim_kwargs = {}
        if self.optimizer == Optimizer.Adam:
            optim_kwargs["eps"] = 1e-5

        optim = self.optimizer(learning_rate=self.lr, **optim_kwargs)
        if self.max_grad_norm is not None:
            optim = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optim,
            )
        return optim


@dataclass(frozen=True, kw_only=True)
class DummyMultiTaskConfig(OptimizerConfig):
    @property
    def requires_split_task_losses(self) -> bool:
        return True

    def spawn(self) -> optax.GradientTransformation:
        return optax.chain(
            dummy_multitask_optimizer(),
            super().spawn(),
        )


@dataclass(frozen=True, kw_only=True)
class PCGradConfig(OptimizerConfig):
    num_tasks: int
    cosine_sim_logs: bool = False

    @property
    def requires_split_task_losses(self) -> bool:
        return True

    def spawn(self) -> optax.GradientTransformationExtraArgs:
        return optax.chain(
            pcgrad(num_tasks=self.num_tasks, cosine_sim_logs=self.cosine_sim_logs),
            super().spawn(),
        )


@dataclass(frozen=True, kw_only=True)
class GradNormConfig(OptimizerConfig):
    num_tasks: int
    gradnorm_optimizer: OptimizerConfig
    initial_weights: jax.Array | None = None
    asymmetry: float = 0.12

    @property
    def requires_split_task_losses(self) -> bool:
        return True

    def spawn(self) -> optax.GradientTransformation:
        return optax.chain(
            gradnorm(
                optim=self.gradnorm_optimizer,
                num_tasks=self.num_tasks,
                asymmetry=self.asymmetry,
                initial_weights=self.initial_weights,
            ),
            super().spawn(),
        )
