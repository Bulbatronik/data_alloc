#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --exclude=node2119,node3006
#SBATCH --output=logs/%x-%j.out
#
# Prereg 19 M3 (expected step-0 gradient + optimiser geometry) for ONE start:
#   1. firstgrad.py --n_batches NB --adam_dirs (additive flags; the Part 1 code path is untouched and regression-proved):
#      NB i.i.d. GRPO batches (8 pool prompts x 8 rollouts, T 0.8, marked + stop reward) at the merged-fresh start; saves
#      v0_lora_g<G> (raw), sign_g<G>, adamv_g<G>, adamv25_g<G>, adam_run1/2 and the matched random controls, the
#      per-batch Gram matrices (m3_gram.*) and the weight-space norms / inner products (m3_adam.json).
#   2. runs/theory/scratch/m3phi.py: exact phi of every such direction on Part 1's registered set A (one per-sample
#      gradient pass), plus Part 1's v0_lora / rand_lora (bridge; its --compare prints FD-vs-exact agreement only).
# Resumable: complete rollout batches, the direction files and finished phi samples are kept on requeue.
# Submit from the repo root AFTER registration, one job per start (4 jobs; <= 4 GPUs on mit_preemptable):
#   for S in 42 43 44; do sbatch --job-name=p19-m3-gold-s$S --export=ALL,START=gold,SEED=$S runs/reviews/round4/scratch/S19_m3.sbatch; done
#   sbatch --job-name=p19-m3-untrained-i42 --export=ALL,START=untrained,SEED=42 runs/reviews/round4/scratch/S19_m3.sbatch
# Variables: START gold|untrained; SEED 42|43|44 (init seed = SEED, as Part 1); NB [256]; GLIST ["1 4 16 64 256"];
#            NRAND [4] (j = 1 is Part 1's rand_lora); OUT [runs/firstgrad/m3]
set -eo pipefail
module load miniforge/25.11.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc
cd /orcd/scratch/orcd/006/usemil/data_alloc
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
START=${START:?gold|untrained}; SEED=${SEED:?42|43|44}
NB=${NB:-256}; GLIST=${GLIST:-1 4 16 64 256}; NRAND=${NRAND:-4}; OUT=${OUT:-runs/firstgrad/m3}
METHOD=random_sft_250; DTAG=gold_s${SEED}_merged_i${SEED}
if [ "$START" = untrained ]; then METHOD=full_grpo; DTAG=untrained_i${SEED}; fi
RUN=runs/gsm8k/Llama-3.2-1B-Instruct-px-mf/seed_$SEED/$METHOD
mkdir -p "$OUT"
echo "=== $(date '+%F %T') M3 directions: $START s$SEED, $NB batches, G {$GLIST}, --adam_dirs"
python firstgrad.py --stage directions --start "$START" --seed "$SEED" --lora_init_seed "$SEED" --run_dir "$RUN" \
  --out_dir "$OUT" --n_batches "$NB" --g_list $GLIST --n_rand "$NRAND" --ckpts 25 150 --adam_dirs \
  --timings_out "$OUT/timings.jsonl"
echo "=== $(date '+%F %T') M3 exact phi on Part 1's set A"
python runs/theory/scratch/m3phi.py --start "$START" --seed "$SEED" --lora_init_seed "$SEED" --dirs_dir "$OUT" \
  --extra_dir runs/firstgrad/part1 --samples_dir runs/firstgrad/part1 \
  --compare "runs/firstgrad/part1/phi_${DTAG}_A.jsonl" --timings_out "$OUT/timings.jsonl"
echo "=== $(date '+%F %T') done"
