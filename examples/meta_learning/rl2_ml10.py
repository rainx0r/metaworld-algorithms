from dataclasses import dataclass
from pathlib import Path

import tyro
import numpy as np

from metaworld_algorithms.config.networks import (
    RecurrentContinuousActionPolicyConfig,
)
from metaworld_algorithms.config.nn import RecurrentNeuralNetworkConfig, VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import (
    GradientBasedMetaLearningTrainingConfig,
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
    evaluation_frequency: int = 1_000_000


def main() -> None:
    args = tyro.cli(Args)

    meta_batch_size = 20
    num_tasks = 10

    run = Run(
        run_name="ml10_rl2_bptt",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldMetaLearningConfig(
            env_id="ML10",
            meta_batch_size=meta_batch_size,
            recurrent_info_in_obs=True,
        ),
        algorithm=RL2Config(
            num_tasks=meta_batch_size,
            meta_batch_size=meta_batch_size,
            gamma=0.995,
            gae_lambda=0.97,
            policy_config=RecurrentContinuousActionPolicyConfig(
                encoder_config=VanillaNetworkConfig(
                    depth=1,
                    width=256,
                    activation=Activation.Tanh,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                ),
                # encoder_config=None,
                network_config=RecurrentNeuralNetworkConfig(
                    width=256,
                    cell_type=CellType.GRU,
                    activation=Activation.Tanh,
                    recurrent_kernel_init=Initializer.ORTHOGONAL,
                    kernel_init=Initializer.XAVIER_UNIFORM,
                    bias_init=Initializer.ZEROS,
                    optimizer=OptimizerConfig(max_grad_norm=1.0)
                ),
                log_std_min=np.log(1e-6),
                log_std_max=None,
                std_type=StdType.MLP_HEAD,
                squash_tanh=False,
                head_kernel_init=Initializer.XAVIER_UNIFORM,
                head_bias_init=Initializer.ZEROS,
            ),
            num_epochs=10,
            chunk_len=250,
            normalize_advantages=False,
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
