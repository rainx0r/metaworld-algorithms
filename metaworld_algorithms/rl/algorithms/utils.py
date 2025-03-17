from typing import Any

import optax
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training.train_state import TrainState as FlaxTrainState


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
