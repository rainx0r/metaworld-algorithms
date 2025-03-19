from metaworld_algorithms.checkpoint import checkpoint_envs
from metaworld_algorithms.envs.metaworld import MetaworldMetaLearningConfig
import chex


def test_ml45_test_envs_spawn_deterministic():
    SEED = 42
    env_config = MetaworldMetaLearningConfig("ML45")

    envs1 = env_config.spawn_test(SEED)
    envs1_ckpt = checkpoint_envs(envs1)

    for _ in range(1):
        envs2 = env_config.spawn_test(SEED)
        envs2_ckpt = checkpoint_envs(envs2)

        for env_ckpt1, env_ckpt2 in zip(envs1_ckpt, envs2_ckpt):
            env_ckpt1_tasks, env_ckpt2_tasks = (
                env_ckpt1[1]["tasks"],
                env_ckpt2[1]["tasks"],
            )
            for task1, task2 in zip(env_ckpt1_tasks, env_ckpt2_tasks):
                assert task1.data == task2.data
