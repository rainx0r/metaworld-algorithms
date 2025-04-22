import enum

import flax.linen
import jax
import optax


def _uniform_init(bound: float) -> jax.nn.initializers.Initializer:
    import metaworld_algorithms.nn.initializers

    return metaworld_algorithms.nn.initializers.uniform(bound)


class Initializer(enum.Enum):
    ZEROS = enum.member(lambda: jax.nn.initializers.zeros)  # noqa: E731
    HE_NORMAL = enum.member(jax.nn.initializers.he_normal)
    HE_UNIFORM = enum.member(jax.nn.initializers.he_uniform)
    XAVIER_NORMAL = enum.member(jax.nn.initializers.xavier_normal)
    XAVIER_UNIFORM = enum.member(jax.nn.initializers.xavier_uniform)
    CONSTANT = enum.member(jax.nn.initializers.constant)
    UNIFORM = enum.member(_uniform_init)

    def __call__(self, *args):
        return self.value(*args)


class Activation(enum.Enum):
    ReLU = enum.member(jax.nn.relu)
    Tanh = enum.member(jax.nn.tanh)
    LeakyReLU = enum.member(jax.nn.leaky_relu)
    PReLU = enum.member(lambda x: flax.linen.PReLU()(x))  # noqa: E731
    ReLU6 = enum.member(jax.nn.relu6)
    SiLU = enum.member(jax.nn.silu)
    GELU = enum.member(jax.nn.gelu)
    GLU = enum.member(jax.nn.glu)
    Identity = enum.member(lambda x: x)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class Optimizer(enum.Enum):
    Adam = enum.member(optax.adam)
    AdamW = enum.member(optax.adamw)
    RMSProp = enum.member(optax.rmsprop)
    SGD = enum.member(optax.sgd)

    def __call__(self, learning_rate: float, **kwargs):
        return self.value(learning_rate, **kwargs)


class StdType(enum.Enum):
    MLP_HEAD = enum.auto()
    PARAM = enum.auto()


class CellType(enum.Enum):
    RNN = enum.member(flax.linen.RNNCellBase)
    LSTM = enum.member(flax.linen.OptimizedLSTMCell)
    GRU = enum.member(flax.linen.GRUCell)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
