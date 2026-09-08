# Reinforcement Learning II · University of Mannheim · Fall 2026

Shared repository for **Reinforcement Learning II 2026** at the University of
Mannheim. This teaching fork builds on
[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), with the Zoo
infrastructure largely unchanged and teaching material under `course/`.
The course starts with its core workflows; you are welcome to explore additional
upstream tools in your own branch.

## Gymnasium, Stable-Baselines3 and RL Zoo

Gymnasium defines the environment interface: observations, actions, rewards and
episode boundaries. Stable-Baselines3 (SB3) provides reinforcement learning
algorithms implemented in PyTorch, including PPO. RL Baselines3 Zoo connects
these algorithms to experiment configuration, training, evaluation, hyperparameter
tuning, plotting and video recording. Environment-specific configurations live
in `hyperparams/`.

References:

- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/)
- [RL Baselines3 Zoo documentation](https://rl-baselines3-zoo.readthedocs.io/en/master/)
- [Gymnasium documentation](https://gymnasium.farama.org/)

## Installation

The course setup has been tested on macOS. Linux and Windows commands are included,
but platform-specific dependencies may require additional setup.

### Prerequisites

- Git to clone the repository and manage changes.
- `uv` to manage Python, the virtual environment and packages. It combines tasks
  commonly handled by `pip` and `venv`, using `pyproject.toml` and `uv.lock` to
  reproduce the project's dependencies.
- A Bash-capable terminal for the training scripts. On Windows, use Git Bash or WSL.
- `ffmpeg` if you later generate videos. It is not needed to play the included
  introductory video in the notebook.

You do not need to install Python separately: `uv` can install the course's
Python version, 3.12.

### Install uv once

Skip this step if `uv --version` already works.

macOS with Homebrew:

```bash
brew install uv
```

macOS or Linux without Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

Open a new terminal if needed, then check:

```bash
uv --version
```

### Install the repository

```bash
git clone https://github.com/Mannheim-Probability/RLII_26.git
cd RLII_26
uv python install 3.12
```

On macOS, use the additional compiler flag for the PyBullet source build:

```bash
CFLAGS="-fno-define-target-os-macros" uv sync --locked --all-extras
```

On Linux and Windows:

```bash
uv sync --locked --all-extras
```

This creates `.venv`, installs the repository in editable mode and installs the
course dependencies: Gymnasium with Box2D, SB3, MuJoCo, PyBullet, plotting and video
tools, Weights & Biases, and the `notebooks` extra with JupyterLab and ipykernel.
The notebook packages are included in `uv.lock`; no separate `pip` installation
is needed. Editable mode means imports use the code in this checkout.

Use `uv run --locked --all-extras ...` to run commands in the project environment
without activating it manually. Keep `--all-extras` when synchronizing so optional
course packages remain installed.

### Verify the installation

Python and lockfile:

```bash
uv run --locked --all-extras python --version
uv lock --check
```

Core and notebook packages:

```bash
uv run --locked --all-extras python -c "import gymnasium, stable_baselines3, rl_zoo3, mujoco, pybullet, wandb, ipykernel, jupyterlab; print('Imports OK')"
```

LunarLander/Box2D:

```bash
uv run --locked --all-extras python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); obs, info = env.reset(seed=0); print(obs.shape, env.action_space); env.close()"
```

MuJoCo:

```bash
uv run --locked --all-extras python -c "import gymnasium as gym; env = gym.make('HalfCheetah-v4'); obs, info = env.reset(seed=0); print(obs.shape); env.close()"
```

PyBullet:

```bash
uv run --locked --all-extras python -c "import gymnasium as gym; import rl_zoo3.import_envs; env = gym.make('HalfCheetahBulletEnv-v0'); obs, info = env.reset(seed=0); print(obs.shape); env.close()"
```

On macOS, PyBullet may print warnings about duplicate SDL classes from OpenCV and
Pygame. If the command prints the observation shape and exits successfully, the
smoke test has passed.

## Lecture 01: explore the repository

Launch the [Lecture 01 notebook](course/lecture_01/lecture_01.ipynb):

```bash
uv run --locked --all-extras jupyter lab course/lecture_01/lecture_01.ipynb
```

In VS Code, select the repository's `.venv` interpreter as the notebook kernel.
See [the notebook guide](course/lecture_01/NOTEBOOK.md) for details. The introductory
video is included at
`course/lecture_01/videos/generated/training_progress/training.mp4` and appears
when you execute the video cell. You do not need to generate it or train a model first.

The [example training script](course/lecture_01/scripts/train_ppo_example.sh)
launches a short PPO run on CartPole through `train.py`:

```bash
uv run --locked --all-extras bash course/lecture_01/scripts/train_ppo_example.sh
```

The notebook assignment asks you to adapt this file for PPO on LunarLander with
one million training timesteps per run and at least three different training seeds.
Use the notebook's flag descriptions and the CLI help to work out the changes:

```bash
uv run --locked --all-extras python train.py --help
```

The final notebook cells read your completed runs and generate one evaluation
plot: the mean across training seeds with a shaded ±1 standard error region.
Training logs, models and resulting plots stay under `logs/` and are not committed.
Clear notebook outputs before committing your own notebook changes.

## Repository overview

- `rl_zoo3/`: RL Baselines3 Zoo experiment infrastructure.
- `hyperparams/`: algorithm and environment configurations.
- `train.py`, `enjoy.py`: top-level training and policy evaluation entry points.
- `scripts/`: upstream plotting, video and experiment utilities.
- `tests/`: upstream tests and course tests as they are added.
- `docs/`: upstream reference documentation.
- `course/`: lecture notebooks, exercises and setup guides.
- `pyproject.toml`, `uv.lock`, `.python-version`: dependencies and Python version.

The upstream `rl-trained-agents` submodule is not needed for the course workflow;
leave it uninitialized.

## Origin and license

This teaching fork started from
[`DLR-RM/rl-baselines3-zoo`](https://github.com/DLR-RM/rl-baselines3-zoo), commit
`bef2e8fda66c792ee3ae733eaa6d22294b00737e`.

The upstream code is distributed under the MIT license; see [LICENSE](LICENSE).
