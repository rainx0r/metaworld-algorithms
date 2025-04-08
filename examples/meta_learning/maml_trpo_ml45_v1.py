from dataclasses import dataclass
from pathlib import Path

import tyro
import numpy as np

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.rl import (
    GradientBasedMetaLearningTrainingConfig,
)
from metaworld_algorithms.config.utils import Activation, Initializer, StdType
from metaworld_algorithms.envs import MetaworldMetaLearningConfig
from metaworld_algorithms.rl.algorithms import MAMLTRPOConfig
from metaworld_algorithms.run import Run


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False
    evaluation_frequency: int = 4_500_000


def main() -> None:
    args = tyro.cli(Args)

    meta_batch_size = 45
    num_tasks = 45

    run = Run(
        run_name="ml45_mamltrpo",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldMetaLearningConfig(
            env_id="ML45",
            meta_batch_size=meta_batch_size,
            reward_func_version="v1",
            total_goals_per_task_test=45,
        ),
        algorithm=MAMLTRPOConfig(
            num_tasks=meta_batch_size,
            meta_batch_size=meta_batch_size,
            gamma=0.99,
            gae_lambda=1.0,
            policy_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    depth=2,
                    width=512,
                    activation=Activation.Tanh,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                ),
                log_std_min=np.log(1e-6),
                log_std_max=None,
                std_type=StdType.PARAM,
                squash_tanh=False,
                head_kernel_init=Initializer.XAVIER_UNIFORM,
                head_bias_init=Initializer.ZEROS,
            ),
        ),
        training_config=GradientBasedMetaLearningTrainingConfig(
            meta_batch_size=meta_batch_size,
            evaluate_on_train=False,
            total_steps=int(2_000_000 * num_tasks),
            evaluation_frequency=args.evaluation_frequency,
        ),
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        run.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=run,
            resume="allow",
        )

    run.start()


if __name__ == "__main__":
    main()
