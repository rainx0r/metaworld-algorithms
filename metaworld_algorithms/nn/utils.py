import flax.linen as nn


def name_prefix(module: nn.Module) -> str:
    return module.name + "_" if module.name else ""
