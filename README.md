# metaworld-algorithms
Implementations of Multi-Task and Meta-Learning baselines for the Metaworld benchmark

## Structure

Here is how you can navigate this repository:

- `examples` contains code for running baselines.
- `metaworld_algorithms/rl/algorithms` contains the implementations of baseline *algorithms* (e.g. MTSAC, MTPPO, MAML, etc).
- `metaworld_algorithms/nn` contains the implementations of *neural network architectures* used in multi-task RL (e.g. Soft-Modules, PaCo, MOORE, etc).
- `metaworld_algorithms/rl/networks.py` contains code that wraps these neural network building blocks into agent components (actor networks, critic networks, etc).
- `metaworld_algorithms/rl/buffers.py` contains code for the buffers used.
- `metaworld_algorithms/rl/algorithms/base.py` contains code for training loops (e.g. on-policy, off-policy, meta-rl).
- `meatworld_algorithms/envsmetaworld.py` contains utilities for wrapping metaworld for use with these baselines.

