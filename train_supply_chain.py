"""Small PPO experiment runner for supply_chain_nons6_jax.py.

Keep this file beside supply_chain_nons6_jax.py. Install:
    python -m pip install "jaxnasium[algs]==0.1.1" jaxon

Examples:
    python train_supply_chain.py --smoke
    python train_supply_chain.py --seeds 0 1 2 --timesteps 500000
    python train_supply_chain.py --write-config experiment.json
    python train_supply_chain.py --config experiment.json --seeds 0 1 2
    python train_supply_chain.py --eval-run "runs/EXPERIMENT_DIRECTORY/seed_0"

Each training seed gets agent.eqx, config.json, versions.json,
training_returns.csv, evaluation_returns.csv, and summary.json.
training_returns.csv contains one row per PPO update. A blank return means
no episode finished during that update, not a failed training run.
Evaluation is deterministic-policy evaluation on fresh environment rollouts;
the separate evaluation seed is shared across training seeds for comparison.

This starter saves/evaluates the final agent. It does not implement RL Zoo's
periodic evaluation, best-model selection, or TensorBoard callback pipeline.
The default MLP/settings are a starting point, not a tuned configuration or
an exact reproduction of SB3 PPO. In particular, optimizer, policy output,
and value-clipping implementations differ between the libraries.

The environment already scales observations and reward (reward / 2000).
Additional running observation/reward normalization is disabled by default;
enable the PPO flags in the JSON config if your experiment needs it.
Compilation is included in the recorded training time. Seeds run sequentially
in one process so equal-shape runs can reuse the compiled training program.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxnasium as jym
from jaxnasium.algorithms import PPO, RLAgent
from jaxnasium.algorithms.architectures import MLP
import numpy as np

from rl_zoo3.custom_envs.supply_chain_nons6_jax import SupplyChainNonS6EnvJax


DEFAULT_CONFIG = {
    "env": {},  # Any SupplyChainNonS6Env constructor arguments go here.
    "network": {"hidden_sizes": [64, 64], "activation": "tanh"},
    "ppo": {
        "total_timesteps": 1000000,
        "num_envs": 16,
        "num_steps": 512,
        "num_minibatches": 8,  # 16 * 256 / 16 = 256 samples per minibatch.
        "num_epochs": 10,
        "learning_rate_start": 0.0003,
        "learning_rate_end": None,  # Constant learning rate.
        "ent_coef_start": 0.0,
        "ent_coef_end": None,
        "gamma": 0.999,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "clip_coef_vf": 10.0,  # Jaxnasium value clipping; not SB3's None.
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_observations": False,
        "normalize_rewards": False,
        "warmup_steps": 0,
        "log_function": "simple",
        "log_interval": 1,
        "reduce_metrics_fn": "mean",
    },
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_env(env_config):
    env = SupplyChainNonS6EnvJax(**env_config)
    return jym.LogWrapper(jym.FlattenObservationWrapper(env))


def make_trainer(config):
    network = config["network"]
    activation = {"tanh": jnp.tanh, "relu": jax.nn.relu}[network["activation"]]
    body = MLP.with_params(hidden_sizes=tuple(network["hidden_sizes"]),
                           activation=activation)
    return PPO(**config["ppo"], actor_kwargs={"body": body},
               critic_kwargs={"body": body})


def evaluate(agent, env, episodes, seed):
    @eqx.filter_jit
    def evaluate_compiled(agent, key):
        return agent.evaluate(key, env, num_eval_episodes=episodes,
                              deterministic_policy=True)

    returns = np.asarray(evaluate_compiled(agent, jax.random.key(seed)))
    if not np.all(np.isfinite(returns)):
        raise RuntimeError("Evaluation produced non-finite returns.")
    return returns


def validate_config(config):
    ppo = config["ppo"]
    for name in ("total_timesteps", "num_envs", "num_steps", "num_minibatches", "num_epochs"):
        value = ppo[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"ppo.{name} must be a positive integer")
    if ppo["warmup_steps"] < 0:
        raise ValueError("ppo.warmup_steps must be >= 0")
    rollout_size = ppo["num_envs"] * ppo["num_steps"]
    if rollout_size % ppo["num_minibatches"] != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    if rollout_size // ppo["num_minibatches"] < 2:
        raise ValueError("Use at least 2 samples per PPO minibatch")
    if ppo["reduce_metrics_fn"] != "mean":
        raise ValueError("This CSV writer requires reduce_metrics_fn='mean'")
    if ppo["log_function"] not in (None, "simple", "tqdm"):
        raise ValueError("log_function must be null, 'simple', or 'tqdm'")
    if ppo["log_interval"] < 1 or int(ppo["log_interval"]) != ppo["log_interval"]:
        raise ValueError("Use a positive integer log_interval (PPO updates)")
    if config["network"]["activation"] not in ("tanh", "relu"):
        raise ValueError("network.activation must be 'tanh' or 'relu'")
    sizes = config["network"]["hidden_sizes"]
    if not sizes or any(not isinstance(n, int) or n <= 0 for n in sizes):
        raise ValueError("network.hidden_sizes must contain positive integers")
    # Round up to a whole rollout, avoiding Jaxnasium's implicit floor division.
    requested = ppo["total_timesteps"]
    ppo["total_timesteps"] = ((requested + rollout_size - 1) // rollout_size) * rollout_size
    return requested


def run_training(config, seed, eval_seed, eval_episodes, directory, requested):
    directory.mkdir(parents=True, exist_ok=False)
    env = make_env(config["env"])
    trainer = make_trainer(config)
    write_json(directory / "config.json", config)
    packages = ("jax", "jaxlib", "jaxnasium", "equinox", "optax", "distrax", "jaxon")
    write_json(directory / "versions.json", {name: metadata.version(name) for name in packages})
    print(f"\nSeed {seed}: {trainer.total_timesteps:,} training steps -> {directory}", flush=True)
    print("Compiling/training; the first run includes JAX compilation.", flush=True)

    started = time.perf_counter()
    agent, metrics = trainer.train(jax.random.key(seed), env)
    jax.block_until_ready((agent, metrics))
    train_seconds = time.perf_counter() - started
    agent.save(str(directory / "agent.eqx"))

    rollout_size = trainer.num_envs * trainer.num_steps
    # Jaxnasium 0.1.1 always collects at least one warmup step per env,
    # even with warmup_steps=0. This is additional to total_timesteps.
    warmup = max(1, trainer.warmup_steps // trainer.num_envs) * trainer.num_envs
    with (directory / "training_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["update", "training_timesteps", "environment_steps_including_warmup",
                         "mean_completed_episode_return"])
        for i, value in enumerate(np.asarray(metrics), start=1):
            writer.writerow([i, i * rollout_size, warmup + i * rollout_size,
                             float(value) if np.isfinite(value) else ""])

    returns = evaluate(agent, env, eval_episodes, eval_seed)
    np.savetxt(directory / "evaluation_returns.csv", returns[:, None], delimiter=",",
               header="episode_return", comments="")
    summary = {
        "training_seed": seed,
        "evaluation_seed": eval_seed,
        "evaluation_episodes": eval_episodes,
        "deterministic_policy": True,
        "requested_training_timesteps": requested,
        "actual_training_timesteps": trainer.total_timesteps,
        "extra_warmup_environment_steps": warmup,
        "training_seconds_including_compilation": train_seconds,
        "evaluation_return_mean": float(returns.mean()),
        "evaluation_return_std": float(returns.std()),
        "evaluation_unscaled_net_reward_mean": float(returns.mean()) * 2000.0,
        "devices": [str(device) for device in jax.devices()],
    }
    write_json(directory / "summary.json", summary)
    print(f"Evaluation return: {returns.mean():.3f} +/- {returns.std():.3f} "
          f"(across {eval_episodes} episodes)", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON config; omitted fields use defaults")
    parser.add_argument("--write-config", type=Path, help="Write an editable default config and exit")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--smoke", action="store_true", help="Small run, retaining the full S6 episode")
    parser.add_argument("--eval-run", type=Path, help="Reload a saved seed directory and evaluate only")
    args = parser.parse_args()

    if args.write_config:
        with args.write_config.open("x") as stream:
            json.dump(DEFAULT_CONFIG, stream, indent=2)
            stream.write("\n")
        print(f"Wrote {args.write_config}")
        return
    if args.eval_episodes <= 0 or args.eval_seed < 0 or any(seed < 0 for seed in args.seeds):
        parser.error("Use positive eval-episodes and nonnegative seeds")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Training seeds must be unique")

    if args.eval_run:
        config = json.loads((args.eval_run / "config.json").read_text())
        agent = RLAgent.load(str(args.eval_run / "agent.eqx"))
        returns = evaluate(agent, make_env(config["env"]), args.eval_episodes, args.eval_seed)
        print(json.dumps({"returns": returns.tolist(), "mean": float(returns.mean()),
                          "std": float(returns.std())}, indent=2))
        return

    # Fail before training if the optional checkpoint dependency is missing.
    try:
        import jaxon  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install checkpoint support: python -m pip install jaxon") from exc

    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        overrides = json.loads(args.config.read_text())
        for section, values in overrides.items():
            if section not in config or not isinstance(values, dict):
                parser.error("Config sections must be env, ppo, or network dictionaries")
            config[section].update(values)
    if args.smoke:
        config["ppo"].update(total_timesteps=8192, num_envs=4, num_steps=256,
                             num_minibatches=4, num_epochs=1)
    if args.timesteps is not None:
        config["ppo"]["total_timesteps"] = args.timesteps
    if args.num_envs is not None:
        config["ppo"]["num_envs"] = args.num_envs
    requested = validate_config(config)
    # Save all environment defaults too, so they cannot silently change on reload.
    config["env"] = dataclasses.asdict(SupplyChainNonS6EnvJax(**config["env"]))
    make_trainer(config)  # Validate constructor kwargs before creating output dirs.

    experiment = args.out / datetime.now(timezone.utc).strftime("ppo_s6_%Y%m%dT%H%M%S_%fZ")
    print(f"JAX devices: {jax.devices()}", flush=True)
    results = [run_training(config, seed, args.eval_seed, args.eval_episodes,
                            experiment / f"seed_{seed}", requested) for seed in args.seeds]
    means = np.array([result["evaluation_return_mean"] for result in results])
    write_json(experiment / "summary.json", {
        "seeds": args.seeds,
        "mean_evaluation_return_across_training_seeds": float(means.mean()),
        "std_of_evaluation_means_across_training_seeds": float(means.std()),
        "runs": results,
    })
    print(f"\nSaved experiment: {experiment}", flush=True)


if __name__ == "__main__":
    main()
