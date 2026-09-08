"""Plot training/evaluation logs, or compare the two five-seed GAE experiments."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import plot_results


def plot_run(run: Path) -> list[tuple[str, plt.Figure]]:
    if not run.is_dir():
        raise FileNotFoundError(f"Run not found: {run}")
    figures = []
    # Reuse SB3's Monitor reader and plotter. Short runs show individual episodes;
    # SB3 adds a 100-episode moving average only when enough episodes exist.
    if list(run.glob("*.monitor.csv")) and not load_results(str(run)).empty:
        plot_results([str(run)], None, "timesteps", "PPO: training return", figsize=(9, 4))
        plt.xlabel("Timesteps in completed training episodes")
        figures.append(("training_return.png", plt.gcf()))

    evaluation_file = run / "evaluations.npz"
    if evaluation_file.is_file():
        with np.load(evaluation_file) as data:
            timesteps = data["timesteps"]
            returns = data["results"]
        if len(timesteps):
            mean, sd = returns.mean(axis=1), returns.std(axis=1)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(timesteps, mean, label="Mean evaluation return")
            ax.fill_between(timesteps, mean - sd, mean + sd, alpha=0.2, label="±1 SD across evaluation episodes")
            ax.set(xlabel="Environment timesteps", ylabel="Episode return", title="PPO: evaluation return")
            ax.legend()
            fig.tight_layout()
            figures.append(("evaluation_return.png", fig))
    if not figures:
        raise ValueError(f"No completed training episodes or evaluations in {run}")
    return figures


def plot_comparison(root: Path) -> list[tuple[str, plt.Figure]]:
    groups = {}
    for lam, directory in [(0.95, "gae_0_95"), (0.98, "gae_0_98")]:
        curves = []
        for seed in range(5):
            # Explicit run 1: a rerun must be selected deliberately, not mixed into the group.
            path = root / directory / f"seed_{seed}" / "ppo" / "LunarLander-v3_1" / "evaluations.npz"
            with np.load(path) as data:
                curves.append(pd.Series(data["results"].mean(axis=1), index=data["timesteps"], name=seed))
        groups[lam] = pd.concat(curves, axis=1).dropna()
    common = groups[0.95].index.intersection(groups[0.98].index).sort_values()
    if common.empty:
        raise ValueError("No evaluation timesteps shared by all ten runs")
    fig, ax = plt.subplots(figsize=(9, 4))
    for lam, frame in groups.items():
        frame = frame.loc[common]
        mean, sd = frame.mean(axis=1), frame.std(axis=1, ddof=1)
        ax.plot(common, mean, label=f"λ={lam}, five seeds")
        ax.fill_between(common, mean - sd, mean + sd, alpha=0.2)
    ax.set(xlabel="Environment timesteps", ylabel="Mean evaluation return", title="Bands: ±1 SD across training seeds")
    ax.legend()
    fig.tight_layout()
    return [("gae_comparison.png", fig)]


class RunArgsLoader(yaml.SafeLoader):
    """Read Zoo's OrderedDict metadata without permitting arbitrary Python objects."""


RunArgsLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:collections.OrderedDict",
    lambda loader, node: dict(loader.construct_sequence(node, deep=True)[0]),
)


def assignment_runs(root: Path) -> dict[int, Path]:
    """Select the latest completed assignment run for each distinct training seed."""
    selected = {}
    for run in root.glob("ppo/LunarLander-v3_*"):
        suffix = run.name.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            continue
        metadata = run / "LunarLander-v3" / "args.yml"
        evaluation = run / "evaluations.npz"
        if not (metadata.is_file() and evaluation.is_file() and (run / "LunarLander-v3.zip").is_file()):
            continue
        with metadata.open() as stream:
            args = yaml.load(stream, Loader=RunArgsLoader)
        if args.get("algo") != "ppo" or args.get("env") != "LunarLander-v3" or args.get("n_timesteps") != 1_000_000:
            continue
        seed = args.get("seed")
        if not isinstance(seed, int) or seed < 0:
            continue
        with np.load(evaluation) as data:
            steps, results = data["timesteps"], data["results"]
            if (steps.ndim != 1 or not len(steps) or steps[-1] < 1_000_000
                    or results.ndim != 2 or results.shape[0] != len(steps)
                    or results.shape[1] == 0 or not np.isfinite(results).all()):
                continue
        if seed not in selected or int(suffix) > int(selected[seed].name.rsplit("_", 1)[-1]):
            selected[seed] = run
    if len(selected) < 3:
        raise ValueError(
            f"Found {len(selected)} completed assignment seeds in {root}. "
            "Finish at least three distinct seeds with the requested environment and budget. "
            "Keep evaluation enabled and wait for the final model to be saved."
        )
    return dict(sorted(selected.items()))


def plot_assignment(runs: dict[int, Path]) -> list[tuple[str, plt.Figure]]:
    curves = []
    fig, ax = plt.subplots(figsize=(10, 5))
    for seed, run in runs.items():
        with np.load(run / "evaluations.npz") as data:
            curve = pd.Series(data["results"].mean(axis=1), index=data["timesteps"], name=seed)
        curves.append(curve)
    shared = pd.concat(curves, axis=1).dropna().sort_index()
    if shared.empty:
        plt.close(fig)
        raise ValueError("No shared evaluation timesteps. Use identical evaluation settings for all seeds.")
    mean = shared.mean(axis=1)
    standard_error = shared.std(axis=1, ddof=1) / np.sqrt(shared.shape[1])
    ax.plot(shared.index, mean, color="black", linewidth=2, label="Mean across seeds")
    ax.fill_between(
        shared.index, mean - standard_error, mean + standard_error,
        color="black", alpha=0.15, label="±1 standard error across seeds",
    )
    ax.set(
        xlabel="Environment timesteps", ylabel="Evaluation episode return",
        title="PPO on LunarLander: mean evaluation return ± standard error",
    )
    ax.legend()
    fig.tight_layout()
    return [("seed_comparison.png", fig)]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--run-dir", type=Path, help="Exact Zoo run directory")
    inputs.add_argument("--comparison", action="store_true", help="Plot the existing two five-seed groups")
    parser.add_argument("--output-dir", type=Path, help="Defaults to a plots directory inside the selected logs")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    if args.comparison:
        source = root / "logs/lecture_01/ppo_lunarlander_gae_comparison"
        figures = plot_comparison(source)
    else:
        source = args.run_dir
        if source is None:
            parent = root / "logs/lecture_01/ppo_assignment/ppo"
            runs = [p for p in parent.glob("LunarLander-v3_*") if p.name.rsplit("_", 1)[-1].isdigit()]
            if not runs:
                parser.error("Adapt and run train_ppo_example.sh first, or supply --run-dir.")
            source = max(runs, key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        figures = plot_run(source)
    print("Data:", source.resolve())
    output = args.output_dir or source / "plots"
    output.mkdir(parents=True, exist_ok=True)
    for filename, fig in figures:
        destination = output / filename
        fig.savefig(destination, dpi=160)
        print("Saved:", destination.resolve())
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
