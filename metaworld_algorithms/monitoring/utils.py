import flax.struct
import numpy.typing as npt
import wandb
from jaxtyping import Float


class Histogram(flax.struct.PyTreeNode):
    data: Float[npt.NDArray, "..."]


def log(logs: dict, step: int) -> None:
    for key, value in logs.items():
        if isinstance(value, Histogram):
            logs[key] = wandb.Histogram(value.data)  # pyright: ignore[reportArgumentType]
    wandb.log(logs, step=step)
