#!/usr/bin/env bash
#SBATCH --job-name=data_alloc
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

# Runs `python run_curve.py <args>` on one GPU. Submitted by run_all.sh.
# If preempted, the job is requeued and finished allocations are skipped.
set -eo pipefail

module load miniforge/25.11.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc

nvidia-smi --query-gpu=name,memory.total --format=csv
python run_curve.py "$@"
