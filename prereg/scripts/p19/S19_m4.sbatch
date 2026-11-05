#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=node2119,node3006
#SBATCH --output=logs/%x-%j.out
#
# Prereg 19 M4 (gradient alignment) via runs/theory/scratch/galign.py for ONE start and a list of sets. Resumable per
# completed item (samples) and per gradient shard, so preemption/requeue only loses the current chunk. Submit from the
# repo root AFTER registration; e.g. split each start into two jobs to use 4 preemptable GPUs:
#   for S in gold untrained; do
#     sbatch --job-name=p19-m4-$S-a --export=ALL,START=$S,SETS="pool gsm8k svamp" runs/reviews/round4/scratch/S19_m4.sbatch
#     sbatch --job-name=p19-m4-$S-b --export=ALL,START=$S,SETS="gsm_plus math600" runs/reviews/round4/scratch/S19_m4.sbatch
#   done
# Variables: START gold|untrained (seed 42, init seed 42); SETS; K [16]; TEMP [0.6]; OUT [runs/galign]
set -eo pipefail
module load miniforge/25.11.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc
cd /orcd/scratch/orcd/006/usemil/data_alloc
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
START=${START:?gold|untrained}; SETS=${SETS:-pool gsm8k gsm_plus svamp math600}
K=${K:-16}; TEMP=${TEMP:-0.6}; OUT=${OUT:-runs/galign}
METHOD=random_sft_250; [ "$START" = untrained ] && METHOD=full_grpo
RUN=runs/gsm8k/Llama-3.2-1B-Instruct-px-mf/seed_42/$METHOD
mkdir -p "$OUT"
echo "=== $(date '+%F %T') galign $START sets {$SETS} K=$K T=$TEMP"
python runs/theory/scratch/galign.py --stage all --start "$START" --seed 42 --lora_init_seed 42 --run_dir "$RUN" \
  --sets $SETS --k "$K" --temperature "$TEMP" --out_dir "$OUT" --timings_out "$OUT/timings.jsonl"
echo "=== $(date '+%F %T') done"
