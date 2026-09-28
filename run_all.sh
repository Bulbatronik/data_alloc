#!/usr/bin/env bash
# Submit one Slurm job per seed (each trains every allocation for that seed):
#   ./run_all.sh
# When all jobs have finished, aggregate + log the W&B curve-summary run:
#   srun -p mit_normal -c 2 --mem=8G -t 00:30:00 ./run_all.sh aggregate
#
# Settings can be overridden from the environment, e.g. a quick smoke test:
#   N_TRAIN=32 N_EVAL=32 SEEDS=42 EXTRA="--output_dir runs/smoke --wandb_group smoke" \
#     PARTITION=mit_normal_gpu TIME=00:45:00 ./run_all.sh
set -eo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
N_TRAIN="${N_TRAIN:-1000}"
N_EVAL="${N_EVAL:--1}"          # -1 = full GSM8K test set (1319)
SEEDS="${SEEDS:-42 43 44}"
COUNTS="${COUNTS:-250 500 750}"   # SFT counts; 0 and N are always added
PARTITION="${PARTITION:-mit_preemptable}"
TIME="${TIME:-48:00:00}"
EXTRA="${EXTRA:-}"

ARGS=(
  --model "$MODEL"
  --max_train_samples "$N_TRAIN"
  --max_eval_samples "$N_EVAL"
  --counts $COUNTS
  --strategies random adaptive
  --bf16
  $EXTRA
)

if [[ "${1:-}" == "aggregate" ]]; then
  module load miniforge/25.11.0-0
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc
  python run_curve.py "${ARGS[@]}" --seeds $SEEDS
  exit 0
fi

mkdir -p logs
for SEED in $SEEDS; do
  sbatch -p "$PARTITION" -t "$TIME" --job-name="alloc-${MODEL##*/}-s$SEED" \
    job.sbatch "${ARGS[@]}" --seeds "$SEED" --no_aggregate
done
