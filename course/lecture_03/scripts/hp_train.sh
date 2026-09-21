#!/usr/bin/env bash
# Start or continue an Optuna study for PPO on LunarLander-v3.
#
# Usage from the repository root:
#   bash course/lecture_03/scripts/hp_train.sh [SAMPLER] [N_TRIALS]
#
# Examples:
#   bash course/lecture_03/scripts/hp_train.sh random 20
#   bash course/lecture_03/scripts/hp_train.sh tpe 20
#   bash course/lecture_03/scripts/hp_train.sh auto 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

# Parameters students will most often change.
SAMPLER="${1:-tpe}"
N_TRIALS="${2:-20}"
PRUNER="${PRUNER:-median}"
STUDY_NAME="${STUDY_NAME:-ppo-lunarlander}"

# Budget for each trial.
TIMESTEPS_PER_TRIAL="${TIMESTEPS_PER_TRIAL:-1000000}"
N_EVALUATIONS="${N_EVALUATIONS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
N_STARTUP_TRIALS="${N_STARTUP_TRIALS:-5}"
N_JOBS="${N_JOBS:-1}"
SEED="${SEED:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/logs/lecture_03/hpo}"
DATABASE="${DATABASE:-$OUTPUT_DIR/ppo_lunarlander.db}"
STORAGE="sqlite:///$DATABASE"

case "$SAMPLER" in
    random|tpe|auto) ;;
    *)
        echo "Unknown sampler: $SAMPLER"
        echo "Choose random, tpe, or auto."
        exit 2
        ;;
esac

case "$PRUNER" in
    none|median|halving) ;;
    *)
        echo "Unknown pruner: $PRUNER"
        echo "Choose none, median, or halving."
        exit 2
        ;;
esac

if [[ ! "$N_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_JOBS must be a positive integer."
    exit 2
fi

if [[ "$PRUNER" == "halving" && "$N_JOBS" -eq 1 ]]; then
    echo "The RL Zoo halving pruner requires parallel trials."
    echo "Use N_JOBS=2 or choose PRUNER=median."
    exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found at: $PYTHON"
    echo "Create the repository environment first with: uv sync --locked --all-extras"
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/matplotlib"
export MPLCONFIGDIR="$OUTPUT_DIR/matplotlib"

echo "Study:              $STUDY_NAME"
echo "Sampler / pruner:   $SAMPLER / $PRUNER"
echo "New trials:         $N_TRIALS"
echo "Parallel trials:    $N_JOBS"
echo "Steps per trial:    $TIMESTEPS_PER_TRIAL"
echo "Study database:     $DATABASE"
echo

"$PYTHON" "$REPO_ROOT/train.py" \
    --algo ppo \
    --env LunarLander-v3 \
    --optimize-hyperparameters \
    --sampler "$SAMPLER" \
    --pruner "$PRUNER" \
    --n-trials "$N_TRIALS" \
    --n-jobs "$N_JOBS" \
    --n-startup-trials "$N_STARTUP_TRIALS" \
    --n-timesteps "$TIMESTEPS_PER_TRIAL" \
    --n-evaluations "$N_EVALUATIONS" \
    --eval-episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --study-name "$STUDY_NAME" \
    --storage "$STORAGE" \
    --log-folder "$OUTPUT_DIR/reports" \
    --no-optim-plots
