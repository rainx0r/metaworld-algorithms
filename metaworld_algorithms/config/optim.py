from dataclasses import dataclass

import optax

from .utils import Optimizer


@dataclass(frozen=True, kw_only=True)
class OptimizerConfig:
    lr: float = 3e-4
    optimizer: Optimizer = Optimizer.Adam
    max_grad_norm: float | None = None

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
