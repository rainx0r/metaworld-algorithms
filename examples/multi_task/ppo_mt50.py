from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    ValueFunctionConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OnPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms import PPOConfig
from metaworld_algorithms.run import Run


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

    num_tasks = 50

    run = Run(
        run_name="mt50_ppo",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(env_id="MT50"),
        algorithm=PPOConfig(
            num_tasks=num_tasks,
            gamma=0.99,
            policy_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            vf_config=ValueFunctionConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            num_epochs=16,
            num_gradient_steps=32,
            gae_lambda=0.97,
            target_kl=None,
            clip_vf_loss=False,
        ),
        training_config=OnPolicyTrainingConfig(
            total_steps=int(1e8),
            rollout_steps=10_000,
            evaluation_frequency=int(1_000_000 // 500),
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
