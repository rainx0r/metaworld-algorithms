import pathlib
import random
from dataclasses import dataclass

import jax
import numpy as np
import orbax.checkpoint as ocp
import tyro

import metaworld_algorithms.checkpoint as ckpt_utils
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.nn import MOOREConfig
from metaworld_algorithms.config.rl import OffPolicyTrainingConfig
from metaworld_algorithms.envs.metaworld import MetaworldConfig
from metaworld_algorithms.rl.algorithms import MTSACConfig, OffPolicyAlgorithm


# CLI args
@dataclass(frozen=True)
class Arguments:
    seed: int = 42
    """Seed to use"""

    checkpoint: bool = True
    """Whether the run should checkpoint."""

    resume: bool = False
    """Whether the run should resume."""

    run_name: str = "MOORE_MT10"
    """The name for the run."""

    enable_wandb: bool = True
    """Whether the run should log to wandb."""


def main(args: Arguments) -> None:
    if jax.device_count("gpu") < 1 and jax.device_count("tpu") < 1:
        raise RuntimeError(
            "No accelerator found, aborting. Devices: %s" % jax.devices()
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    data_dir = pathlib.Path(f"runs/{args.run_name}")
    data_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: Configure the env
    env_config = MetaworldConfig(
        env_id="MT10",
        reward_func_version="v2",
    )
    envs = env_config.spawn(args.seed)

    # NOTE: Configure the algorithm
    num_tasks = 10
    algorithm = MTSACConfig(
        num_tasks=num_tasks,
        actor_config=ContinuousActionPolicyConfig(
            network_config=MOOREConfig(num_tasks=num_tasks)
        ),
        critic_config=QValueFunctionConfig(
            network_config=MOOREConfig(num_tasks=num_tasks)
        ),
        use_task_weights=True,
    ).spawn(env_config, args.seed)
    training_config = OffPolicyTrainingConfig(
        total_steps=20_000_000,
        warmstart_steps=4_000,
        buffer_size=1_000_000,
    )

    # NOTE: Run

    # Checkpointing setup boilerplate
    buffer_checkpoint = None
    checkpoint_manager = None
    checkpoint_metadata = None
    envs_checkpoint = None
    if args.checkpoint:
        checkpoint_items = (
            "agent",
            "env_states",
            "rngs",
            "metadata",
            "buffer",
        )

        checkpoint_manager = ocp.CheckpointManager(
            (data_dir / "checkpoints").absolute(),
            item_names=checkpoint_items,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=5,
                create=True,
                best_fn=lambda x: x["mean_success_rate"],
            ),
        )

        if args.resume and checkpoint_manager.latest_step() is not None:
            assert isinstance(algorithm, OffPolicyAlgorithm)
            rb = algorithm.spawn_replay_buffer(
                env_config,
                training_config,
            )
            ckpt: ckpt_utils.Checkpoint = checkpoint_manager.restore(  # pyright: ignore [reportAssignmentType]
                checkpoint_manager.latest_step(),
                args=ckpt_utils.get_checkpoint_restore_args(algorithm, rb),
            )
            algorithm = ckpt["agent"]
            buffer_checkpoint = ckpt["buffer"]  # pyright: ignore [reportTypedDictNotRequiredAccess]

            envs_checkpoint = ckpt["env_states"]
            ckpt_utils.load_env_checkpoints(envs, envs_checkpoint)

            random.setstate(ckpt["rngs"]["python_rng_state"])
            np.random.set_state(ckpt["rngs"]["global_numpy_rng_state"])

            checkpoint_metadata: ckpt_utils.CheckpointMetadata | None = ckpt["metadata"]
            assert checkpoint_metadata is not None

            print(f"Loaded checkpoint at step {checkpoint_metadata['step']}")

    _wandb = None
    if args.enable_wandb:
        import wandb

        wandb.init(dir=str(data_dir), id=args.run_name, name=args.run_name)
        _wandb = wandb

    # The actual run
    agent = algorithm.train(
        config=training_config,
        envs=envs,
        env_config=env_config,
        track=args.enable_wandb,
        checkpoint_manager=checkpoint_manager,
        checkpoint_metadata=checkpoint_metadata,
        buffer_checkpoint=buffer_checkpoint,
    )

    # Final eval
    mean_success_rate, mean_returns, mean_success_per_task = env_config.evaluate(
        envs, agent
    )
    final_metrics = {
        "mean_success_rate": float(mean_success_rate),
        "mean_evaluation_return": float(mean_returns),
    } | {
        f"{task_name}_success_rate": float(success_rate)
        for task_name, success_rate in mean_success_per_task.items()
    }
    print("Final metrics", final_metrics)

    if args.checkpoint:
        assert checkpoint_manager is not None
        checkpoint_manager.save(
            training_config.total_steps + 1,
            args=ckpt_utils.get_last_agent_checkpoint_save_args(agent, final_metrics),
            metrics=final_metrics,
        )
        checkpoint_manager.wait_until_finished()

        if args.enable_wandb:
            assert _wandb is not None
            # Log final model checkpoint
            assert _wandb.run is not None
            final_ckpt_artifact = _wandb.Artifact(
                f"{_wandb.run.id}_final_agent_checkpoint", type="model"
            )
            final_ckpt_dir = checkpoint_manager._get_save_directory(
                training_config.total_steps + 1, checkpoint_manager.directory
            )
            final_ckpt_artifact.add_dir(str(final_ckpt_dir))
            _wandb.log_artifact(final_ckpt_artifact)

            # Log best model checkpoint (by mean success rate)
            best_step = checkpoint_manager.best_step()
            assert best_step is not None
            best_ckpt_artifact = _wandb.Artifact(
                f"{_wandb.run.id}_best_agent_checkpoint", type="model"
            )
            best_ckpt_dir = checkpoint_manager._get_save_directory(
                best_step, checkpoint_manager.directory
            )
            best_ckpt_artifact.add_dir(str(best_ckpt_dir))
            _wandb.log_artifact(best_ckpt_artifact)

        checkpoint_manager.close()

    print("Run finished!")


if __name__ == "__main__":
    main(tyro.cli(Arguments))
