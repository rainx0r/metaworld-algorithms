from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from metaworld_algorithms.config.nn import PaCoConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OffPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.run import Run
from metaworld_algorithms.rl.algorithms import MTSACConfig


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

    run = Run(
        run_name="mt50_paco",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT50",
            terminate_on_success=False,
        ),
        algorithm=MTSACConfig(
            num_tasks=50,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=PaCoConfig(
                    num_tasks=50,
                    num_parameter_sets=20,
                    optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=PaCoConfig(
                    num_tasks=50,
                    num_parameter_sets=20,
                    optimizer=OptimizerConfig(max_grad_norm=1.0), 
                )
            ),
            num_critics=2,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(1e8),
            buffer_size=int(100_000 * 50),
            batch_size=6400,
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
