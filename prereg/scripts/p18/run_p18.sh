#!/bin/bash
# score_prereg18.py on CPU via Slurm only. Usage: sbatch [opts] run_p18.sh <args to score_prereg18.py>
module load miniforge/25.11.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc
cd /orcd/scratch/orcd/006/usemil/data_alloc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_DATASETS_OFFLINE=1
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname)"
python -I runs/theory/scratch/score_prereg18.py "$@"
