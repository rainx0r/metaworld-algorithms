from dataclasses import dataclass
from pathlib import Path

import tyro
# import jax

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    ValueFunctionConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import GradientBasedMetaLearningTrainingConfig, MetaLearningTrainingConfig
from metaworld_algorithms.envs import MetaworldMetaLearningConfig
from metaworld_algorithms.rl.algorithms import MAMLTRPOConfig
from metaworld_algorithms.run import Run

# jax.config.update("jax_debug_nans", True)


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    meta_batch_size = 20

    run = Run(
        run_name="ml10_mamltrpo",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldMetaLearningConfig(
            env_id="ML10",
        ),
        algorithm=MAMLTRPOConfig(
            num_tasks=meta_batch_size,
            gamma=0.99,
            policy_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            vf_config=ValueFunctionConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            clip_vf_loss=False,
        ),
        training_config=GradientBasedMetaLearningTrainingConfig(
            meta_batch_size=meta_batch_size,
            evaluate_on_train=False,
            total_steps=15_000_000,
            evaluation_frequency=1_000_000,
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
