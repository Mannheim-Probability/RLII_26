#!/usr/bin/env bash
# Activate .venv first, then run this script with bash.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"

for seed in 0 1 2; do
  "${PYTHON:-python}" train.py \
    --algo ppo \
    --env LunarLander-v3 \
    --n-timesteps 1000000 \
    --seed "$seed" \
    --eval-freq 10000 \
    --eval-episodes 5 \
    --n-eval-envs 1 \
    --hyperparams n_envs:16 n_steps:1024 batch_size:64 n_epochs:2 \
    --uuid \
    -f logs/lecture_01/ppo_assignment &
done

wait