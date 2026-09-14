#!/usr/bin/env bash
# Activate .venv first, then run this script with bash.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
mkdir -p "$REPO_ROOT/logs/lecture_02/demo"
cd "$REPO_ROOT/logs/lecture_02/demo"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

for seed in 0 1; do
  "${PYTHON:-python}" "$REPO_ROOT/train.py" \
    --algo ppo \
    --env LunarLander-v3 \
    --n-timesteps 1000000 \
    --seed "$seed" \
    --log-interval 1 \
    --eval-freq 20480 \
    --eval-episodes 3 \
    --n-eval-envs 1 \
    --save-freq 40960 \
    --device cpu \
    --num-threads 1 \
    --hyperparams n_envs:4 n_steps:256 batch_size:64 n_epochs:2 \
    --track \
    --wandb-project-name Lecture_02_demo \
    --wandb-entity RL2_2026 \
    --wandb-group lecture02_demo \
    --uuid \
    -f .
done
