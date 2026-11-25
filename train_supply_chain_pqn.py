"""PQN experiment runner for the JAX supply-chain environment.

Run from your project root (so rl_zoo3.custom_envs is importable). Install:
    python -m pip install "jaxnasium[algs]==0.1.1" jaxon

Examples:
    python train_supply_chain_pqn.py --smoke --eval-episodes 2
    python train_supply_chain_pqn.py --seeds 0 1 2 --timesteps 500000
    python train_supply_chain_pqn.py --write-config experiment_pqn.json
    python train_supply_chain_pqn.py --config experiment_pqn.json --seeds 0 1 2
    python train_supply_chain_pqn.py --eval-run "runs/EXPERIMENT_DIRECTORY/seed_0"

The default environment class preserves your project import:
    rl_zoo3.custom_envs.supply_chain_nons6_jax:SupplyChainNonS6EnvJax
For the standalone environment file from the earlier conversion, use:
    python train_supply_chain_pqn.py --smoke \
        --env-class supply_chain_nons6_jax:SupplyChainNonS6Env

PQN learns Q-values for discrete actions. This runner adapts the S6 Box(0, 1)
order action to Discrete(max_order + 1): action index i orders i units.
For fewer actions, use --order-step 20 (0, 20, 40, ..., max_order; the maximum
is always included). The grid is saved and reconstructed for evaluation.
The adapter encodes each quantity inside its continuous action interval so
float32 rounding before integer truncation cannot accidentally order one less
unit at the usual supply-chain scales. It checks every mapping on construction.
An already-Discrete Jaxnasium environment is passed through unchanged.

Each training seed gets agent.eqx, config.json, versions.json,
training_returns.csv, evaluation_returns.csv, action_quantities.json, and
summary.json. training_returns.csv contains one row per PQN rollout/update.
A blank return means
no episode finished during that update, not a failed training run.
Evaluation is deterministic-policy evaluation on fresh environment rollouts;
the separate evaluation seed is shared across training seeds for comparison.

This starter saves/evaluates the final agent. It does not implement RL Zoo's
periodic evaluation, best-model selection, or TensorBoard callback pipeline.
The MLP/settings are a starting point, not a tuned configuration. PQN uses
one Q-network (critic_kwargs), q_lambda, and epsilon-greedy exploration.

The environment already scales observations and reward (reward / 2000).
Additional running observation/reward normalization is disabled by default;
enable the PQN flags in the JSON config if your experiment needs it.
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
import importlib
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxnasium as jym
from jaxnasium.algorithms import PQN, RLAgent
from jaxnasium.algorithms.architectures import MLP
import numpy as np

DEFAULT_CONFIG = {
    "env_class": "rl_zoo3.custom_envs.supply_chain_nons6_jax:SupplyChainNonS6EnvJax",
    "env": {},  # Your environment constructor arguments go here.
    "actions": {"order_step": 1},  # 1 preserves every integer order quantity.
    "network": {"hidden_sizes": [64, 64], "activation": "tanh"},
    "pqn": {
        "total_timesteps": 100000,
        "num_envs": 16,
        "num_steps": 256,
        "num_minibatches": 16,  # 16 * 256 / 16 = 256 samples per minibatch.
        "num_epochs": 4,
        "learning_rate_start": 0.0003,
        "learning_rate_end": None,  # Constant learning rate.
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "exploration_fraction": 0.5,
        "gamma": 0.999,
        "q_lambda": 0.95,
        "max_grad_norm": 10.0,
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


def load_env_class(reference):
    """Resolve the configured module:class; no silent fallback to another env."""
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("env_class must be a string 'module.path:ClassName'")
    module_name, class_name = reference.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(cls, type) or not issubclass(cls, jym.Environment):
        raise TypeError("env_class must refer to a Jaxnasium Environment subclass")
    return cls


class DiscreteOrderWrapper(jym.Wrapper):
    """Map discrete action indices to exact S6 integer order quantities.

    The underlying S6 environment computes int(action[0] * max_order).
    A half-unit offset moves non-endpoint quantities away from floating-point
    truncation boundaries. No-order and maximum-order map exactly to 0 and 1.
    """

    quantities: tuple[int, ...] = eqx.field(static=True)
    max_order: int = eqx.field(static=True)

    @property
    def action_space(self):
        return jym.Discrete(len(self.quantities), dtype=jnp.int32)

    def continuous_action(self, action):
        action = jnp.asarray(action)
        if action.shape != () or not jnp.issubdtype(action.dtype, jnp.integer):
            raise ValueError("PQN action must be a scalar integer index")
        index = jnp.clip(action, 0, len(self.quantities) - 1)
        quantity = jnp.asarray(self.quantities, jnp.int32)[index]
        fraction = (quantity.astype(jnp.float32) + 0.5) / max(self.max_order, 1)
        fraction = jnp.where(quantity == 0, 0.0,
                             jnp.where(quantity == self.max_order, 1.0, fraction))
        return fraction.astype(jnp.float32).reshape((1,))

    def step(self, key, state, action):
        return self._env.step(key, state, self.continuous_action(action))

    def step_env(self, key, state, action):
        return self._env.step_env(key, state, self.continuous_action(action))


def adapt_actions(env, order_step):
    if not isinstance(order_step, int) or isinstance(order_step, bool) or order_step < 1:
        raise ValueError("actions.order_step must be a positive integer")
    space = env.action_space
    if isinstance(space, jym.Discrete):
        if order_step != 1:
            raise ValueError("order_step only applies to the S6 continuous order action")
        return env
    if not (isinstance(space, jym.Box) and space.shape == (1,)
            and np.all(np.asarray(space.low) == 0.0)
            and np.all(np.asarray(space.high) == 1.0)):
        raise ValueError("Expected S6 Box(0, 1, shape=(1,)) or a Discrete action space")
    max_order = int(env.max_order)
    if max_order < 0 or max_order > np.iinfo(np.int32).max:
        raise ValueError("max_order must fit a nonnegative int32")
    quantities = tuple(range(0, max_order + 1, order_step))
    if quantities[-1] != max_order:
        quantities += (max_order,)
    wrapper = DiscreteOrderWrapper(env, quantities=quantities, max_order=max_order)
    # Check the exact expression used by the S6 step, including JIT fusion.
    @jax.jit
    def encoded_orders(indices):
        fractions = jax.vmap(wrapper.continuous_action)(indices)[:, 0]
        return (jnp.clip(fractions, 0.0, 1.0) * max_order).astype(jnp.int32)
    actual = np.asarray(encoded_orders(jnp.arange(len(quantities), dtype=jnp.int32)))
    if not np.array_equal(actual, np.asarray(quantities, dtype=np.int32)):
        raise ValueError("Order quantities cannot be represented reliably with float32 actions")
    return wrapper


def make_env(config):
    env = load_env_class(config["env_class"])(**config["env"])
    env = adapt_actions(env, config["actions"]["order_step"])
    return jym.LogWrapper(jym.FlattenObservationWrapper(env))


def make_trainer(config):
    network = config["network"]
    activation = {"tanh": jnp.tanh, "relu": jax.nn.relu}[network["activation"]]
    body = MLP.with_params(hidden_sizes=tuple(network["hidden_sizes"]),
                           activation=activation)
    return PQN(**config["pqn"], critic_kwargs={"body": body})


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
    pqn = config["pqn"]
    for name in ("total_timesteps", "num_envs", "num_steps", "num_minibatches", "num_epochs"):
        value = pqn[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"pqn.{name} must be a positive integer")
    if pqn["warmup_steps"] < 0:
        raise ValueError("pqn.warmup_steps must be >= 0")
    for name in ("gamma", "q_lambda", "epsilon_start"):
        if not 0.0 <= pqn[name] <= 1.0:
            raise ValueError(f"pqn.{name} must be between 0 and 1")
    if pqn["epsilon_end"] is not None and not 0.0 <= pqn["epsilon_end"] <= 1.0:
        raise ValueError("pqn.epsilon_end must be between 0 and 1, or null")
    if not 0.0 < pqn["exploration_fraction"] <= 1.0:
        raise ValueError("pqn.exploration_fraction must be in (0, 1]")
    order_step = config["actions"]["order_step"]
    if not isinstance(order_step, int) or isinstance(order_step, bool) or order_step <= 0:
        raise ValueError("actions.order_step must be a positive integer")
    rollout_size = pqn["num_envs"] * pqn["num_steps"]
    if rollout_size % pqn["num_minibatches"] != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    if rollout_size // pqn["num_minibatches"] < 2:
        raise ValueError("Use at least 2 samples per PQN minibatch")
    if pqn["reduce_metrics_fn"] != "mean":
        raise ValueError("This CSV writer requires reduce_metrics_fn='mean'")
    if pqn["log_function"] not in (None, "simple", "tqdm"):
        raise ValueError("log_function must be null, 'simple', or 'tqdm'")
    if pqn["log_interval"] < 1 or int(pqn["log_interval"]) != pqn["log_interval"]:
        raise ValueError("Use a positive integer log_interval (PQN updates)")
    if config["network"]["activation"] not in ("tanh", "relu"):
        raise ValueError("network.activation must be 'tanh' or 'relu'")
    sizes = config["network"]["hidden_sizes"]
    if not sizes or any(not isinstance(n, int) or n <= 0 for n in sizes):
        raise ValueError("network.hidden_sizes must contain positive integers")
    # Round up to a whole rollout, avoiding Jaxnasium's implicit floor division.
    requested = pqn["total_timesteps"]
    pqn["total_timesteps"] = ((requested + rollout_size - 1) // rollout_size) * rollout_size
    return requested


def run_training(config, seed, eval_seed, eval_episodes, directory, requested):
    directory.mkdir(parents=True, exist_ok=False)
    env = make_env(config)
    trainer = make_trainer(config)
    write_json(directory / "config.json", config)
    quantities = None
    current = env
    while isinstance(current, jym.Wrapper):
        if isinstance(current, DiscreteOrderWrapper):
            quantities = list(current.quantities)
            break
        current = current._env
    write_json(directory / "action_quantities.json", {
        "num_actions": env.action_space.n,
        "quantity_by_action_index": quantities,
        "note": "Native Discrete environment" if quantities is None else "Integer order quantities",
    })
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
        "algorithm": "PQN",
        "num_actions": env.action_space.n,
        "order_step": config["actions"]["order_step"],
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
    parser.add_argument("--env-class", help="Environment class as module.path:ClassName")
    parser.add_argument("--order-step", type=int, help="Integer order spacing; default 1 keeps all quantities")
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
        if args.order_step is not None:
            parser.error("Evaluation uses the saved action grid; do not pass --order-step")
        if args.env_class is not None:
            config["env_class"] = args.env_class
        agent = RLAgent.load(str(args.eval_run / "agent.eqx"))
        if not isinstance(agent.trainer, PQN):
            parser.error("--eval-run must point to a PQN checkpoint")
        returns = evaluate(agent, make_env(config), args.eval_episodes, args.eval_seed)
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
            if section == "env_class" and isinstance(values, str):
                config[section] = values
            elif section in ("env", "network", "actions", "pqn") and isinstance(values, dict):
                config[section].update(values)
            else:
                parser.error("Use env/network/actions/pqn dictionaries and an optional env_class string")
    if args.env_class is not None:
        config["env_class"] = args.env_class
    if args.order_step is not None:
        config["actions"]["order_step"] = args.order_step
    if args.smoke:
        config["pqn"].update(total_timesteps=8192, num_envs=4, num_steps=256,
                             num_minibatches=4, num_epochs=1)
    if args.timesteps is not None:
        config["pqn"]["total_timesteps"] = args.timesteps
    if args.num_envs is not None:
        config["pqn"]["num_envs"] = args.num_envs
    requested = validate_config(config)
    # Save all environment defaults too, so they cannot silently change on reload.
    config["env"] = dataclasses.asdict(load_env_class(config["env_class"])(**config["env"]))
    make_trainer(config)  # Validate constructor kwargs before creating output dirs.

    experiment = args.out / datetime.now(timezone.utc).strftime("pqn_s6_%Y%m%dT%H%M%S_%fZ")
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
