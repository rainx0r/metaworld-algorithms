from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    ValueFunctionConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.rl import (
    GradientBasedMetaLearningTrainingConfig,
)
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
    evaluation_frequency: int = 1_000_000


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
                network_config=VanillaNetworkConfig(),
                squash_tanh=False,
            ),
            vf_config=ValueFunctionConfig(network_config=VanillaNetworkConfig()),
            gae_lambda=1.0,
        ),
        training_config=GradientBasedMetaLearningTrainingConfig(
            meta_batch_size=meta_batch_size,
            evaluate_on_train=False,
            total_steps=15_000_000,
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
