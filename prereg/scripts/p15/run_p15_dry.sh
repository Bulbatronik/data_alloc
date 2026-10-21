#!/bin/bash
# Dry runs of score_prereg15.py (CPU, Slurm only). Usage: sbatch/srun run_p15_dry.sh <mode> [extra args]
#   mode: smoke | power | synthetic | e2 | schema_<part>
module load miniforge/25.11.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc
cd /orcd/scratch/orcd/006/usemil/data_alloc
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname)"
S=runs/theory/scratch/score_prereg15.py
mode=$1; shift
case $mode in
  smoke)
    python -I -m py_compile runs/theory/scratch/p15lib.py $S && echo COMPILED
    python -I $S --dry_run power --power_reps 4 2>&1 | tail -40
    python -I $S --dry_run synthetic --reps 3 --null_reps 1 --nsim 8 --cov_reps 2 --cov_B 10 --reps16 2 2>&1 | tail -150
    python -I $S --dry_run e2 --B 20 --nsim 8 2>&1 | tail -150 ;;
  power|power_check|synthetic|e2) export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; python -I $S --dry_run $mode "$@" ;;
  schema_*) python -I $S --part ${mode#schema_} --schema_only "$@" ;;
  mock)  # end-to-end test of every --part on a MOCK root (synthetic correctness, real schemas and golds)
    MR=runs/theory/scratch/p15_mockroot
    rm -rf $MR && python -I runs/theory/scratch/p15_mock.py $MR && \
    python -I $S --root $MR --part all --unblind "$@" ;;
esac
