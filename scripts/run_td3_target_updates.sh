#!/usr/bin/env bash

# Compare TD3 and IC-TD3 on LunarLanderContinuous-v3 with paired seeds.
# Run with bash scripts/run_td3_target_updates.sh after logging in to W&B.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
N_TIMESTEPS="${N_TIMESTEPS:-300000}"
EVAL_FREQ="${EVAL_FREQ:-25000}"
WANDB_ENTITY="${WANDB_ENTITY:-RL2_2026}"
WANDB_GROUP="${WANDB_GROUP:-lunarlander-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/logs/target_updates_td3/$WANDB_GROUP}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$MPLCONFIGDIR"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_ROOT}"
if [[ "$(uname -s)" == Darwin ]]; then
  # W&B 0.28.2 can crash while reading network statistics on sandboxed macOS.
  export WANDB_X_DISABLE_STATS="${WANDB_X_DISABLE_STATS:-true}"
fi
cd "$REPO_ROOT"

pids=()
stop_jobs() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}
trap 'stop_jobs; exit 130' INT
trap 'stop_jobs; exit 143' TERM

failed=0
for algo in td3 ic_td3; do
  pids=()
  run_logs=()

  for seed in 0 1 2 3 4; do
    run_dir="$OUTPUT_ROOT/$algo/seed_$seed"
    mkdir -p "$run_dir"
    run_log="$run_dir/train.log"
    printf 'Starting %s seed %s. Log: %s\n' "$algo" "$seed" "$run_log"

    "$PYTHON" -u -m rl_zoo3.train \
      --algo "$algo" --env LunarLanderContinuous-v3 \
      --seed "$seed" --n-timesteps "$N_TIMESTEPS" --eval-freq "$EVAL_FREQ" \
      --device cpu --num-threads 1 --log-folder "$run_dir" \
      --track --wandb-entity "$WANDB_ENTITY" --wandb-project-name target_updates_td3 \
      --wandb-group "$WANDB_GROUP" --wandb-tags "$algo" lunarlander-continuous \
      --hyperparams gamma:0.99 >"$run_log" 2>&1 &
    pids+=("$!")
    run_logs+=("$run_log")
  done

  # Finish this batch before starting the next so at most five jobs run at once.
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      printf 'Training failed. See %s\n' "${run_logs[$index]}" >&2
      failed=1
    fi
    unset 'pids[index]'
  done
done

if (( failed )); then
  exit 1
fi
printf 'Completed all ten runs. Results: %s\n' "$OUTPUT_ROOT"
