#!/usr/bin/env bash
# Submit prereg/20 S1: Llama-3.2-3B-Instruct, GSM8K, LoRA r16 (alpha 32, dropout 0.05, all linear layers), merged-fresh
# protocol (prereg 15 Part X): gold-250, teacher-250 (Qwen2.5-7B-I traces, runs/gsm8k/teachers/qwen7b.jsonl) and RL alone,
# seeds 42-44. Run ONLY after prereg/20 is committed and pushed (the scorer refuses --unblind otherwise).
#
# 1. SFT (--skip_grpo): pipeline defaults, as for the 1B gold / teacher adapters of 08 S' (lr 1e-4, 1 epoch, batch 8,
#    1,024 tokens); same 1,000-problem pool (--data_seed 1234) and the same seed-S 250/750 split as the 1B runs (the
#    partition depends only on the data and the seed; the 1B and 3B tokenizers and chat templates are byte-identical).
# 2. GRPO from base + merged SFT LoRA with a fresh LoRA (--merge_sft_start --lora_init_seed S; RL alone gets the same A),
#    375 steps, constant lr 2e-5 after 10 warm-up steps, 8 prompts x 8 samples, T 0.8, top-p 0.95, 512 tokens, beta 0,
#    loss grpo, reward marked + stop, rollouts logged, checkpoints every 75 steps (75/225/375 scored).
#    Smoke job 27672218: micro-batch 8 peaks at 39.4 GB on an L40S (45 GiB); 10 steps at 12-42 s (mean 22.4 s).
#    If a run OOMs, rerun it FROM SCRATCH with --grpo_batch_size 4 --grpo_grad_accum 16 (same 64-completion step and
#    the same generation batch; do not switch micro-batch on a resume).
# 3. probe.py evaluations, T 0.6, top-p 0.95, 512 tokens, all 1,319 test items, marked `correct` field:
#    untrained k = 16; after SFT on the MERGED start (probe --merged_start, the exact GRPO start) k = 4;
#    checkpoints 75 / 225 / 375 of all 9 runs k = 4.
#
# Every step is a CHAIN of RETRY+1 jobs linked by afterany on mit_preemptable (--requeue): route_sft_grpo.py skips
# finished methods and resumes GRPO from its last checkpoint; probe.py skips adapters already in its --out file.
# Downstream steps wait on afterok of the LAST job of each upstream chain.
# Estimated GPU time (smoke 27672218; see runs/reviews/round4/scratch/T20_scale_report.md):
#   SFT 6 x ~0.15 h ~ 1 h; GRPO 9 x (375 x 20-30 s + ~0.2 h) ~ 21-33 h; evals ~195k samples at ~11k / GPU-h ~ 18 h.
set -euo pipefail
cd /orcd/scratch/orcd/006/usemil/data_alloc
M=unsloth/Llama-3.2-3B-Instruct
G=runs/gsm8k/Llama-3.2-3B-Instruct
OUT=runs/probes/prereg20_3b
RETRY=${RETRY:-2}
mkdir -p logs $OUT
CENV='set -eo pipefail; module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; export PYTHONPATH=. HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1;'
ENV="$CENV"' export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python -c "import torch; assert torch.cuda.is_available()"; nvidia-smi --query-gpu=name --format=csv,noheader;'
GPU="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_preemptable -t 12:00:00 --requeue --output=logs/%x-%j.out"

# chain NAME DEP CMD -> submits CMD and RETRY afterany retries; echoes the last job id
chain() {
  local name=$1 dep=$2; shift 2
  local d=$dep j
  for ((i = 0; i <= RETRY; i++)); do
    j=$(sbatch $GPU ${d:+--dependency=$d} --job-name=$name-r$i --wrap "$ENV $*")
    d="afterany:$j"
  done
  echo "$name: last job $j" >&2
  echo "$j"
}
TEACH="--sft_traces runs/gsm8k/teachers/qwen7b.jsonl"
BASE="python route_sft_grpo.py --model $M --dataset gsm8k --bf16 --max_eval_samples 300"   # in-job greedy check on 300 items
RL="$BASE --reward marked --reward_needs_stop --log_rollouts --grpo_steps 375 --grpo_lr_scheduler constant_with_warmup --grpo_warmup_steps 10 --grpo_save_steps 75 --grpo_loss_type grpo --merge_sft_start"

# 1. SFT (gold, then teacher) per seed
declare -A SFT
for S in 42 43 44; do
  SFT[$S]=$(chain p20-3b-sft-s$S "" "$BASE --seed $S --skip_grpo --methods random_sft_250 --output_dir $G-p20-sft/seed_$S --wandb_tags prereg20,S1,sft && $BASE --seed $S --skip_grpo $TEACH --methods random_sft_250 --output_dir $G-p20-sft-qwen7b/seed_$S --wandb_tags prereg20,S1,sft")
done
SFT_ALL="afterok:${SFT[42]}:${SFT[43]}:${SFT[44]}"

# 2. GRPO, merged-fresh: gold, teacher, RL alone (separate output dirs: no shared results.csv between parallel jobs)
TRAIN=()
for S in 42 43 44; do
  TRAIN+=($(chain p20-3b-gold-s$S "afterok:${SFT[$S]}" "$RL --seed $S --lora_init_seed $S --reuse_from $G-p20-sft/seed_$S --methods random_sft_250 --output_dir $G-p20-mf/seed_$S --wandb_tags prereg20,S1,gold"))
  TRAIN+=($(chain p20-3b-teacher-s$S "afterok:${SFT[$S]}" "$RL --seed $S --lora_init_seed $S $TEACH --reuse_from $G-p20-sft-qwen7b/seed_$S --methods random_sft_250 --output_dir $G-p20-mf-qwen7b/seed_$S --wandb_tags prereg20,S1,teacher"))
  TRAIN+=($(chain p20-3b-rl-s$S "" "$RL --seed $S --lora_init_seed $S --methods full_grpo --output_dir $G-p20-mf-rl/seed_$S --wandb_tags prereg20,S1,rl"))
done
ALL="afterok:$(IFS=:; echo "${TRAIN[*]}")"

# 3. evaluations
PROBE="python probe.py --model $M --dataset gsm8k --reward marked --temperature 0.6 --top_p 0.95 --max_new_tokens 512 --batch_size 128"
chain p20-3b-ev-unt "" "$PROBE --k 16 --out $OUT/untrained.jsonl" > /dev/null
chain p20-3b-ev-sft "$SFT_ALL" "$PROBE --k 4 --merged_start --out $OUT/sft.jsonl --adapters '$G-p20-sft/seed_*/random_sft_250/sft' '$G-p20-sft-qwen7b/seed_*/random_sft_250/sft'" > /dev/null
for c in 75 225 375; do
  chain p20-3b-ev-s$c "$ALL" "$PROBE --k 4 --out $OUT/s$c.jsonl --adapters '$G-p20-mf*/seed_*/*/grpo/checkpoint-$c'" > /dev/null
done
echo "submitted; afterwards (CPU Slurm): python -I runs/theory/scratch/score_prereg20.py --design 3b --schema_only, then --unblind"
