#!/usr/bin/env bash

# Compare fixed and increasing hard target intervals on LunarLanderContinuous-v3.
# Run with bash scripts/run_td3_hard_targets.sh after logging in to W&B.

set -eo pipefail

dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
  dry_run=true
  shift
fi
if (( $# != 0 )); then
  printf 'Usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
N_TIMESTEPS="${N_TIMESTEPS:-500000}"
EVAL_FREQ="${EVAL_FREQ:-25000}"
WANDB_ENTITY="${WANDB_ENTITY:-RL2_2026}"
WANDB_GROUP="${WANDB_GROUP:-lunarlander-hard-targets-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
WANDB_TAG="td3-hard-targets-adam-sgd-v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/logs/target_updates_td3/$WANDB_GROUP}"

if ! "$dry_run"; then
  if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$PYTHON" >&2
    exit 1
  fi

  export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
  mkdir -p "$OUTPUT_ROOT" "$MPLCONFIGDIR"
  OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
  export WANDB_DIR="${WANDB_DIR:-$OUTPUT_ROOT}"
  if [[ "$(uname -s)" == Darwin ]]; then
    # W&B 0.28.2 can crash while reading network statistics on sandboxed macOS.
    export WANDB_X_DISABLE_STATS="${WANDB_X_DISABLE_STATS:-true}"
  fi
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
for schedule in fixed geometric; do
  if [[ "$schedule" == fixed ]]; then
    algo=td3_target_interval
    interval_key=target_update_interval
    schedule_params=()
  else
    algo=ic_td3
    interval_key=initial_target_update_interval
    schedule_params=(geometric_target_update_intervals:True)
  fi

  for interval in 2 100 1000 10000 50000; do
    for optimizer in adam sgd; do
      if [[ "$optimizer" == adam ]]; then
        policy_kwargs='dict(net_arch=[400, 300], optimizer_class=th.optim.Adam)'
      else
        policy_kwargs='dict(net_arch=[400, 300], optimizer_class=th.optim.SGD, optimizer_kwargs=dict(momentum=0.0))'
      fi

      condition="$schedule-k$interval-$optimizer"
      pids=()
      run_logs=()
      for seed in 0 1 2 3 4; do
        run_dir="$OUTPUT_ROOT/$condition/seed_$seed"
        run_log="$run_dir/train.log"
        command=(
          "$PYTHON" -u -m rl_zoo3.train
          --algo "$algo" --env LunarLanderContinuous-v3
          --seed "$seed" --n-timesteps "$N_TIMESTEPS" --eval-freq "$EVAL_FREQ" --eval-episodes 5
          --device cpu --num-threads 1 --log-folder "$run_dir"
          --track --wandb-entity "$WANDB_ENTITY" --wandb-project-name target_updates_td3
          --wandb-group "$WANDB_GROUP-$condition" --wandb-tags "$WANDB_TAG" --wandb-exact-tags
          --hyperparams tau:1.0 gamma:0.99 policy_delay:2 learning_rate:0.001
          "$interval_key:$interval" "${schedule_params[@]}" "policy_kwargs:'$policy_kwargs'"
        )

        if "$dry_run"; then
          printf '%q ' "${command[@]}"
          printf '\n'
          continue
        fi

        mkdir -p "$run_dir"
        printf 'Starting %s seed %s. Log: %s\n' "$condition" "$seed" "$run_log"
        "${command[@]}" >"$run_log" 2>&1 &
        pids+=("$!")
        run_logs+=("$run_log")
      done

      # Finish all five seeds before starting the next configuration.
      for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
          printf 'Training failed. See %s\n' "${run_logs[$index]}" >&2
          failed=1
        fi
        unset 'pids[index]'
      done
    done
  done
done

if (( failed )); then
  exit 1
fi
if ! "$dry_run"; then
  printf 'Completed all 100 runs. Results: %s\n' "$OUTPUT_ROOT"
fi
