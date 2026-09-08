#!/usr/bin/env bash
# Activate .venv first, then run this script with bash.
# Adapt the environment, training budget and seed for the notebook assignment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"

# Short, runnable example. Keep the output folder and evaluation settings for the assignment.
"${PYTHON:-python}" train.py \
  --algo ppo \
  --env CartPole-v1 \
  --n-timesteps 10000 \
  --seed 0 \
  --eval-freq 10000 \
  --eval-episodes 5 \
  --n-eval-envs 1 \
  --hyperparams n_envs:16 n_steps:1024 batch_size:64 n_epochs:2 \
  -f logs/lecture_01/ppo_assignment
