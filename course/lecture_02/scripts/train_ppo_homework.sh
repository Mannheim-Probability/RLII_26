#!/usr/bin/env bash
# Activate .venv first. Complete the local homework harness before running this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
mkdir -p "$REPO_ROOT/logs/lecture_02/homework"
cd "$REPO_ROOT/logs/lecture_02/homework"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

"${PYTHON:-python}" "$REPO_ROOT/course/lecture_02/scripts/train.py" \
  --algo ppo_homework \
  --conf-file "$REPO_ROOT/hyperparams/ppo.yml" \
  --env LunarLander-v3 \
  --n-timesteps 1000000 \
  --seed 0 \
  --eval-freq 10000 \
  --eval-episodes 5 \
  --n-eval-envs 1 \
  --device cpu \
  --num-threads 1 \
  --hyperparams n_envs:16 n_steps:1024 batch_size:64 n_epochs:2 gamma:0.999 normalize_advantage:False target_kl:None discounting:"'none'" \
  --track \
  --wandb-project-name RL2_2026 \
  --wandb-entity RL2_2026 \
  --wandb-group lecture02_homework \
  --uuid \
  -f .
