#!/bin/bash

set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

if ! command -v uv >/dev/null 2>&1; then
  printf 'Install uv on the Helix login node and make it available on PATH.\n' >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT="$repo/.venv" uv sync --locked --all-extras --python "$(cat .python-version)"
.venv/bin/python - <<'PY'
import gymnasium as gym
import stable_baselines3 as sb3
import torch
import wandb

from rl_zoo3 import ALGOS

env = gym.make("LunarLanderContinuous-v3")
try:
    env.reset(seed=0)
    env.step(env.action_space.sample())
finally:
    env.close()

assert {"td3_target_interval", "ic_td3"} <= ALGOS.keys()
print(f"SB3 {sb3.__version__}; PyTorch {torch.__version__}; CUDA build {torch.version.cuda}; W&B {wandb.__version__}")
print("LunarLanderContinuous-v3 and both TD3 variants are available. The pilot verifies GPU access on its allocated node.")
PY
