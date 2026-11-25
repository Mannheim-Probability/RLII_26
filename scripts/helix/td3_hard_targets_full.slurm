#!/bin/bash
#SBATCH --job-name=td3-v2-full
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-199%5
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
exec bash scripts/helix/run_td3_gpu_task.sh full "$@"
