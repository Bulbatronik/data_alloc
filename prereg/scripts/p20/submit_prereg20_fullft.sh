#!/usr/bin/env bash
# Submit prereg/20 S2: Llama-3.2-1B-Instruct, GSM8K, FULL fine-tuning (no LoRA) for SFT and GRPO: gold-250, teacher-250
# (Qwen2.5-7B-I traces) and RL alone, seeds 42-44. Run ONLY after prereg/20 is committed and pushed.
#
# Full FT uses fp32 master weights (--fp32_master: weights loaded in float32, trained under bf16 autocast), as PEFT keeps
# LoRA weights in fp32. The pipeline's old full-FT path trains pure-bf16 weights, which round away most Adam updates at
# these learning rates (smoke 27672219, 5 GRPO steps at lr 2e-6 from the same SFT weights: 0.50% of weight elements
# changed, ||dW||/||W|| 3.8e-6, vs 87.6% and 4.1e-5 with fp32 master weights).
# Learning rates: LoRA 1e-4 (SFT) / 2e-5 (GRPO) divided by 10 (the pipeline's "LoRA needs ~10x full-FT LR"):
# SFT 1e-5 (= the earlier full-FT runs, runs/gsm8k/Llama-3.2-1B-Instruct-fullft), GRPO 2e-6 (those runs used 1e-6 in
# pure bf16 and did not move: grpo_gain -0.0015 at seed 42).
# 1. SFT (--skip_grpo), otherwise the pipeline defaults (1 epoch, batch 8), same split as the 1B LoRA seeds.
# 2. GRPO from the saved fp32 SFT weights (reuse of a full-model dir); RL alone from the base. 375 steps, constant lr after
#    10 warm-up steps, 8 x 8, T 0.8, top-p 0.95, 512 tokens, beta 0, loss grpo, marked + stop reward, rollouts logged.
#    Checkpoints every 75 steps; --grpo_keep_ckpt_steps 75 225 375 --grpo_ckpt_bf16 keeps only the scored ones, in bf16,
#    without optimizer state, once a newer checkpoint exists (the newest stays complete for resuming): ~10 GB per run
#    at the end (3 checkpoints, 2.47 GB each; the duplicate final model grpo/model.safetensors = checkpoint-375 is
#    deleted after the run) instead of ~79 GB (5 x 14.8 GB + 4.9 GB). DISK: S2 ends at ~96 GB (+ ~15 GB per active run).
#    Smoke 27672219: no gradient checkpointing needed (peak 38.2 GB of 45 GiB); 10 steps at 10-31 s (mean 20.6 s);
#    resume from checkpoint-5 verified.
# 3. probe.py (bf16, as every probe), T 0.6, top-p 0.95, 512 tokens, 1,319 items, k = 4: after SFT (the saved SFT
#    weights) and checkpoints 75 / 225 / 375. Untrained start = the pooled 1B cell of prereg 15 Part A (same base model,
#    prompt and sampling settings): no new untrained evaluation.
set -euo pipefail
cd /orcd/scratch/orcd/006/usemil/data_alloc
M=unsloth/Llama-3.2-1B-Instruct
G=runs/gsm8k/Llama-3.2-1B-Instruct
OUT=runs/probes/prereg20_fullft
RETRY=${RETRY:-2}
mkdir -p logs $OUT
CENV='set -eo pipefail; module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; export PYTHONPATH=. HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1;'
ENV="$CENV"' export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python -c "import torch; assert torch.cuda.is_available()"; nvidia-smi --query-gpu=name --format=csv,noheader;'
GPU="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_preemptable -t 12:00:00 --requeue --output=logs/%x-%j.out"

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
BASE="python route_sft_grpo.py --model $M --dataset gsm8k --bf16 --max_eval_samples 300 --no_lora --fp32_master"
RL="$BASE --grpo_lr 2e-6 --reward marked --reward_needs_stop --log_rollouts --grpo_steps 375 --grpo_lr_scheduler constant_with_warmup --grpo_warmup_steps 10 --grpo_save_steps 75 --grpo_loss_type grpo --grpo_keep_ckpt_steps 75 225 375 --grpo_ckpt_bf16"

declare -A SFT
for S in 42 43 44; do
  SFT[$S]=$(chain p20-ft-sft-s$S "" "$BASE --sft_lr 1e-5 --seed $S --skip_grpo --methods random_sft_250 --output_dir $G-p20-ft-sft/seed_$S --wandb_tags prereg20,S2,sft && $BASE --sft_lr 1e-5 --seed $S --skip_grpo $TEACH --methods random_sft_250 --output_dir $G-p20-ft-sft-qwen7b/seed_$S --wandb_tags prereg20,S2,sft")
done
SFT_ALL="afterok:${SFT[42]}:${SFT[43]}:${SFT[44]}"

TRAIN=()
for S in 42 43 44; do
  TRAIN+=($(chain p20-ft-gold-s$S "afterok:${SFT[$S]}" "$RL --seed $S --reuse_from $G-p20-ft-sft/seed_$S --methods random_sft_250 --output_dir $G-p20-ft/seed_$S --wandb_tags prereg20,S2,gold && rm -f $G-p20-ft/seed_$S/random_sft_250/grpo/model.safetensors"))
  TRAIN+=($(chain p20-ft-teacher-s$S "afterok:${SFT[$S]}" "$RL --seed $S $TEACH --reuse_from $G-p20-ft-sft-qwen7b/seed_$S --methods random_sft_250 --output_dir $G-p20-ft-qwen7b/seed_$S --wandb_tags prereg20,S2,teacher && rm -f $G-p20-ft-qwen7b/seed_$S/random_sft_250/grpo/model.safetensors"))
  TRAIN+=($(chain p20-ft-rl-s$S "" "$RL --seed $S --methods full_grpo --output_dir $G-p20-ft-rl/seed_$S --wandb_tags prereg20,S2,rl && rm -f $G-p20-ft-rl/seed_$S/full_grpo/grpo/model.safetensors"))
done
ALL="afterok:$(IFS=:; echo "${TRAIN[*]}")"

PROBE="python probe.py --model $M --dataset gsm8k --reward marked --temperature 0.6 --top_p 0.95 --max_new_tokens 512 --batch_size 128"
chain p20-ft-ev-sft "$SFT_ALL" "$PROBE --k 4 --out $OUT/sft.jsonl --adapters '$G-p20-ft-sft/seed_*/random_sft_250/sft' '$G-p20-ft-sft-qwen7b/seed_*/random_sft_250/sft'" > /dev/null
for c in 75 225 375; do
  chain p20-ft-ev-s$c "$ALL" "$PROBE --k 4 --out $OUT/s$c.jsonl --adapters '$G-p20-ft/seed_*/*/grpo/checkpoint-$c' '$G-p20-ft-qwen7b/seed_*/*/grpo/checkpoint-$c' '$G-p20-ft-rl/seed_*/*/grpo/checkpoint-$c'" > /dev/null
done
echo "submitted; afterwards (CPU Slurm): python -I runs/theory/scratch/score_prereg20.py --design fullft --schema_only, then --unblind"
