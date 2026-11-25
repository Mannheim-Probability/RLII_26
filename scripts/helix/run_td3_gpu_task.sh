#!/bin/bash

set -euo pipefail

cuda_module="${HELIX_CUDA_MODULE:-devel/cuda}"
if [[ "$cuda_module" != none ]]; then
  module load "$cuda_module"
fi

if [[ -n "${HELIX_PYTHON:-}" ]]; then
  python="$HELIX_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  python="$(pwd)/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  python="$VIRTUAL_ENV/bin/python"
else
  python=python
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export WANDB_MODE="${WANDB_MODE:-online}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec "$python" -u -m scripts.helix.td3_hard_targets "$@"
