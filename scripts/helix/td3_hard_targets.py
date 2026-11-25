"""Run one task of the Helix hard-target TD3 experiment."""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

ENV_ID = "LunarLanderContinuous-v3"
PROJECT = "target_updates_td3"
TAG = "td3-hard-targets-adam-sgd-v2"
FULL_STEPS = 5_000_000
PILOT_STEPS = 500_000
EVAL_FREQ = 25_000
INTERVALS = (2, 100, 1_000, 10_000, 50_000)
ALGORITHMS = ("td3_target_interval", "ic_td3")
OPTIMIZERS = ("adam", "sgd")


def training_command(
    mode: str,
    task_id: int,
    job_id: str,
    output: Path,
    pilot_optimizer: str,
    pilot_steps: int = PILOT_STEPS,
) -> tuple[list[str], Path, str]:
    is_pilot = mode == "pilot"
    intervals, optimizers = ((1_000,), (pilot_optimizer,)) if is_pilot else (INTERVALS, OPTIMIZERS)
    matrix = list(product(ALGORITHMS, intervals, optimizers, range(10)))
    if not 0 <= task_id < len(matrix):
        raise ValueError(f"Task ID {task_id} is outside 0-{len(matrix) - 1} for {mode}")
    algo, interval, optimizer, seed = matrix[task_id]
    tag = f"{TAG}-pilot" if is_pilot else TAG
    condition = f"{algo}-k{interval}-{optimizer}"
    run_dir = output / mode / job_id / condition / f"seed_{seed}"
    policy = (
        "dict(net_arch=[400, 300], optimizer_class=th.optim.Adam)"
        if optimizer == "adam"
        else "dict(net_arch=[400, 300], optimizer_class=th.optim.SGD, optimizer_kwargs=dict(momentum=0.0))"
    )
    schedule = (
        [f"target_update_interval:{interval}"]
        if algo == "td3_target_interval"
        else [f"initial_target_update_interval:{interval}", "geometric_target_update_intervals:True"]
    )
    eval_options = (
        ["--eval-episodes", "5"] if is_pilot else ["--eval-episodes", "16", "--n-eval-envs", "4", "--vec-env", "subproc"]
    )
    args = [
        sys.executable,
        "-u",
        "-m",
        "rl_zoo3.train",
        "--algo",
        algo,
        "--env",
        ENV_ID,
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--num-threads",
        os.getenv("SLURM_CPUS_PER_TASK", "4"),
        "--n-timesteps",
        str(pilot_steps if is_pilot else FULL_STEPS),
        "--eval-freq",
        str(EVAL_FREQ),
        *eval_options,
        "--log-folder",
        str(run_dir / "training"),
        "--track",
        "--wandb-project-name",
        PROJECT,
        "--wandb-entity",
        os.getenv("WANDB_ENTITY", "RL2_2026"),
        "--wandb-group",
        f"{tag}-{job_id}-{condition}",
        "--wandb-exact-tags",
        "--wandb-tags",
        tag,
        "--hyperparams",
        "tau:1.0",
        "gamma:0.99",
        "policy_delay:2",
        "learning_rate:0.001",
        *schedule,
        f"policy_kwargs:'{policy}'",
    ]
    return args, run_dir, algo


def check_environment() -> None:
    from importlib.metadata import version

    import gymnasium as gym
    import numpy as np
    import torch
    from packaging.version import Version
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: F401

    from rl_zoo3 import ALGOS

    versions = {name: version(name) for name in ("stable-baselines3", "torch", "gymnasium", "wandb", "tensorboard")}
    if not Version("2.9.0") <= Version(versions["stable-baselines3"]) < Version("3"):
        raise RuntimeError(f"This branch requires SB3 >=2.9,<3, found {versions['stable-baselines3']}")
    if not all(algo in ALGOS for algo in ALGORITHMS):
        raise RuntimeError("The TD3 variants could not be imported from this checkout")
    print(f"Python {sys.version.split()[0]} at {sys.executable}; packages {versions}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"PyTorch cannot use CUDA (built for CUDA {torch.version.cuda})")
    device = torch.device("cuda:0")
    matrix = torch.ones((32, 32), device=device)
    result = matrix @ matrix
    torch.cuda.synchronize(device)
    if result[0, 0].item() != 32:
        raise RuntimeError("The CUDA smoke operation produced an unexpected result")
    print(f"CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(device)}; CUDA smoke operation passed")

    env = gym.make(ENV_ID)
    try:
        obs, _ = env.reset(seed=0)
        next_obs, reward, *_ = env.step(env.action_space.sample())
        if not np.isfinite(obs).all() or not np.isfinite(next_obs).all() or not math.isfinite(reward):
            raise RuntimeError(f"{ENV_ID} returned nonfinite values in the initial transition")
    finally:
        env.close()
    print(f"{ENV_ID} reset and step passed")


def check_pilot(run_dir: Path, algo: str, elapsed: float, pilot_steps: int = PILOT_STEPS) -> dict:
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from rl_zoo3 import ALGOS

    (checkpoint,) = (run_dir / "training").rglob(f"{ENV_ID}.zip")
    model = ALGOS[algo].load(checkpoint, device="cpu", buffer_size=1)
    if model.num_timesteps != pilot_steps or model._n_target_updates == 0:
        raise RuntimeError("The pilot checkpoint lacks its expected training progress or target updates")
    if any(not torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
        raise RuntimeError("The pilot checkpoint contains nonfinite network parameters")

    (events_file,) = (run_dir / "runs").rglob("events.out.tfevents.*")
    events = EventAccumulator(str(events_file), size_guidance={"scalars": 0}).Reload()
    for key in (
        "train/critic_loss",
        "train/actor_loss",
        "train/target_update_interval",
        "train/n_target_updates",
        "eval/mean_reward",
    ):
        values = events.Scalars(key)
        if not values or any(not math.isfinite(item.value) for item in values):
            raise RuntimeError(f"The pilot has missing or nonfinite scalar values for {key}")
    evaluations = events.Scalars("eval/mean_reward")
    expected_steps = list(range(EVAL_FREQ, pilot_steps + 1, EVAL_FREQ))
    if [event.step for event in evaluations] != expected_steps:
        raise RuntimeError("The pilot evaluation steps do not match the requested budget and evaluation frequency")
    if not any((run_dir / "wandb").glob("*/run-*.wandb")):
        raise RuntimeError("The pilot did not write a local W&B run record")

    first, last = evaluations[0], evaluations[-1]
    steady_seconds = last.wall_time - first.wall_time
    if steady_seconds <= 0:
        raise RuntimeError("Pilot evaluations do not have increasing wall times")
    throughput = (last.step - first.step) / steady_seconds
    return {
        "algorithm": algo,
        "seed": model.seed,
        "env_steps": pilot_steps,
        "wall_seconds": round(elapsed, 2),
        "env_steps_per_second": round(throughput, 2),
        "projected_full_hours": round((elapsed + (FULL_STEPS - pilot_steps) / throughput) / 3600, 2),
        "target_updates": model._n_target_updates,
        "eval_mean_reward": [event.value for event in evaluations],
        "wandb_mode": os.environ["WANDB_MODE"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "full"))
    parser.add_argument("--task-id", type=int, default=os.getenv("SLURM_ARRAY_TASK_ID"))
    parser.add_argument("--pilot-optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--pilot-steps", type=int, default=PILOT_STEPS, help="Environment steps per pilot task")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.task_id is None:
        parser.error("Provide --task-id when running outside a Slurm array")
    if args.mode == "pilot" and args.pilot_steps < 2 * EVAL_FREQ:
        parser.error(f"--pilot-steps must be at least {2 * EVAL_FREQ} for two evaluations")

    repo = Path(__file__).resolve().parents[2]
    output = Path(os.getenv("HELIX_OUTPUT_ROOT") or repo / "logs" / PROJECT / "helix-v2").expanduser().resolve()
    job_id = os.getenv("SLURM_ARRAY_JOB_ID", "manual")
    command, run_dir, algo = training_command(args.mode, args.task_id, job_id, output, args.pilot_optimizer, args.pilot_steps)
    print(shlex.join(command))
    if args.dry_run:
        return

    check_environment()
    if os.environ.setdefault("WANDB_MODE", "online") not in {"online", "offline"}:
        raise ValueError("Use WANDB_MODE=online or offline to record this experiment")
    run_dir.mkdir(parents=True)
    os.environ["WANDB_DIR"] = str(run_dir)
    os.environ["MPLCONFIGDIR"] = str(run_dir / ".matplotlib")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo), os.getenv("PYTHONPATH"))))
    started = time.monotonic()
    subprocess.run(command, cwd=run_dir, check=True)
    elapsed = time.monotonic() - started
    print(f"Training and evaluation finished in {elapsed:.2f} s")
    if args.mode == "pilot":
        result = check_pilot(run_dir, algo, elapsed, args.pilot_steps)
        result_path = run_dir / "pilot_result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Pilot checks passed: {result_path}\n{json.dumps(result)}")


if __name__ == "__main__":
    main()
