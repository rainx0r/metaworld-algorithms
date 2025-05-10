from typing import Any, TYPE_CHECKING

import flax.struct
import flax.traverse_util
import jax.numpy as jnp
import numpy.typing as npt
import wandb
from jaxtyping import Array, Float, PyTree

if TYPE_CHECKING:
    from metaworld_algorithms.types import LogDict


class Histogram(flax.struct.PyTreeNode):
    data: Float[npt.NDArray | Array, "..."] | None = None
    np_histogram: tuple | None = None


def log(logs: dict, step: int) -> None:
    for key, value in logs.items():
        if isinstance(value, Histogram):
            logs[key] = wandb.Histogram(value.data, np_histogram=value.np_histogram)  # pyright: ignore[reportArgumentType]
    wandb.log(logs, step=step)


def get_logs(
    name: str,
    data: Float[npt.NDArray | Array, "..."],
    axis: int | None = None,
    hist: bool = True,
    std: bool = True,
) -> "LogDict":
    ret: "LogDict" = {
        f"{name}_mean": jnp.mean(data, axis=axis),
        f"{name}_min": jnp.min(data, axis=axis),
        f"{name}_max": jnp.max(data, axis=axis),
    }
    if std:
        ret[f"{name}_std"] = jnp.std(data, axis=axis)
    if hist:
        ret[f"{name}"] = Histogram(data.reshape(-1))

    return ret


def prefix_dict(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def pytree_histogram(pytree: PyTree, bins: int = 64) -> dict[str, Histogram]:
    flat_dict = flax.traverse_util.flatten_dict(pytree, sep="/")
    ret = {}
    for k, v in flat_dict.items():
        if isinstance(v, tuple):  # For activations
            v = v[0]
        ret[k] = Histogram(np_histogram=jnp.histogram(v, bins=bins))  # pyright: ignore[reportArgumentType]
    return ret
