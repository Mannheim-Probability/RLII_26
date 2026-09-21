#!/usr/bin/env bash
# Inspect an existing study and optionally train one selected trial again.
#
# Show all trials and the best trial:
#   bash course/lecture_03/scripts/hp_load.sh
#
# Load the hyperparameters from trial 7 and train a fresh PPO agent:
#   bash course/lecture_03/scripts/hp_load.sh 7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
OPTUNA="${OPTUNA:-$REPO_ROOT/.venv/bin/optuna}"

STUDY_NAME="${STUDY_NAME:-ppo-lunarlander}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/logs/lecture_03/hpo}"
DATABASE="${DATABASE:-$OUTPUT_DIR/ppo_lunarlander.db}"
STORAGE="sqlite:///$DATABASE"

if [[ ! -f "$DATABASE" ]]; then
    echo "Study database not found: $DATABASE"
    echo "Run hp_train.sh first."
    exit 2
fi

if [[ ! -x "$OPTUNA" ]]; then
    echo "Optuna command not found at: $OPTUNA"
    exit 2
fi

mkdir -p "$OUTPUT_DIR/matplotlib"
export MPLCONFIGDIR="$OUTPUT_DIR/matplotlib"

echo "Study overview"
echo "=============="
"$OPTUNA" studies --storage "$STORAGE" --format table

echo
echo "Best trial"
echo "=========="
"$OPTUNA" best-trial \
    --storage "$STORAGE" \
    --study-name "$STUDY_NAME" \
    --format table \
    --flatten

echo
echo "All trials"
echo "=========="
"$OPTUNA" trials \
    --storage "$STORAGE" \
    --study-name "$STUDY_NAME" \
    --format table \
    --flatten

# With no trial number, this script only inspects the study.
if [[ $# -eq 0 ]]; then
    echo
    echo "To train a fresh agent with one trial's hyperparameters, run:"
    echo "  bash course/lecture_03/scripts/hp_load.sh TRIAL_NUMBER"
    exit 0
fi

TRIAL_ID="$1"
if [[ ! "$TRIAL_ID" =~ ^[0-9]+$ ]]; then
    echo "Trial number must be a non-negative integer."
    exit 2
fi

FINAL_TIMESTEPS="${FINAL_TIMESTEPS:-1000000}"
FINAL_SEED="${FINAL_SEED:-0}"
FINAL_EVAL_FREQ="${FINAL_EVAL_FREQ:-10000}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-10}"
FINAL_DIR="${FINAL_DIR:-$REPO_ROOT/logs/lecture_03/final}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found at: $PYTHON"
    exit 2
fi

mkdir -p "$FINAL_DIR"

echo
echo "Training a new agent with the parameters from trial $TRIAL_ID"
echo "The Optuna study stores hyperparameters, not trained PPO weights."

"$PYTHON" "$REPO_ROOT/train.py" \
    --algo ppo \
    --env LunarLander-v3 \
    --n-timesteps "$FINAL_TIMESTEPS" \
    --seed "$FINAL_SEED" \
    --study-name "$STUDY_NAME" \
    --storage "$STORAGE" \
    --trial-id "$TRIAL_ID" \
    --eval-freq "$FINAL_EVAL_FREQ" \
    --eval-episodes "$FINAL_EVAL_EPISODES" \
    --n-eval-envs 1 \
    --log-folder "$FINAL_DIR" \
    --uuid
