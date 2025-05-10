from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from metaworld_algorithms.config.networks import (
    RecurrentContinuousActionPolicyConfig,
)
from metaworld_algorithms.config.nn import (
    RecurrentNeuralNetworkConfig,
    VanillaNetworkConfig,
)
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import (
    RNNBasedMetaLearningTrainingConfig,
)
from metaworld_algorithms.config.utils import Activation, CellType, Initializer, StdType
from metaworld_algorithms.envs import MetaworldMetaLearningConfig
from metaworld_algorithms.rl.algorithms import RL2Config
from metaworld_algorithms.run import Run


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False
    evaluation_frequency: int = 100_000
    env_name: str | None = None


def main() -> None:
    args = tyro.cli(Args)

    meta_batch_size = 10

    run = Run(
        run_name=f"ml1_rl2_v1_{args.env_name}",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldMetaLearningConfig(
            env_id="ML1",
            env_name=args.env_name,
            reward_func_version="v1",
            meta_batch_size=meta_batch_size,
            recurrent_info_in_obs=True,
            total_goals_per_task_train=50,
            total_goals_per_task_test=50,
        ),
        algorithm=RL2Config(
            num_tasks=meta_batch_size,
            meta_batch_size=meta_batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            policy_config=RecurrentContinuousActionPolicyConfig(
                encoder_config=VanillaNetworkConfig(
                    width=256,
                    depth=2,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                    activation=Activation.Tanh,
                ),
                network_config=RecurrentNeuralNetworkConfig(
                    width=256,
                    cell_type=CellType.GRU,
                    activation=Activation.Tanh,
                    recurrent_kernel_init=Initializer.ORTHOGONAL,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                    optimizer=OptimizerConfig(lr=5e-4, max_grad_norm=1.0),
                ),
                log_std_min=np.log(1e-6),
                log_std_max=None,
                std_type=StdType.MLP_HEAD,
                squash_tanh=False,
                head_kernel_init=Initializer.XAVIER_UNIFORM,
                head_bias_init=Initializer.ZEROS,
                activate_head=True,
            ),
            num_epochs=10,
            chunk_len=250,
            normalize_advantages=False,
        ),
        training_config=RNNBasedMetaLearningTrainingConfig(
            meta_batch_size=meta_batch_size,
            evaluate_on_train=False,
            total_steps=2_000_000,
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
