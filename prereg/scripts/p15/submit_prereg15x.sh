#!/usr/bin/env bash
# Submit prereg/15 Part X: protocol 2x2 {gold-250, teacher-250 (qwen7b)} x {continuing adapter (cont), merged SFT +
# fresh LoRA (mf)} plus the merged-fresh RL-alone arm, seeds 42-44: 15 GRPO runs in 12 jobs (the gold mf job also runs
# full_grpo, which under --merge_sft_start gets the same fresh_lora init as the mf arms of that seed).
# Run ONLY after prereg/15 is committed and pushed.
#
# Settings = E2's exact command line (prereg 13; W&B run 4w56j5ir = job 25633697, read from
# wandb/run-20261012_054457-4w56j5ir/files/wandb-metadata.json) with --censor removed:
#   150 GRPO steps, linear decay after 10 warm-up steps, checkpoints every 25, marked + stop reward, --log_rollouts,
#   the three panel prefixes, loss grpo, evaluation T 0.6 k 4 on all 1,319 test items (untrained model, after SFT and
#   final, as in E2). GRPO rollouts use the pipeline defaults E2 used (T 0.8, top-p 0.95, 512 tokens, lr 2e-5, KL 0).
# Teacher arms add --sft_traces runs/gsm8k/teachers/qwen7b.jsonl, as the runs that trained those adapters did
# (08 S'; W&B run l6fbde5d), so the SFT / GRPO split is reproduced exactly.
# Outputs (the scorer's default --px_template):
#   runs/gsm8k/Llama-3.2-1B-Instruct-px-{cont,mf}[-qwen7b]/seed_S/random_sft_250/{predictions_after_sft,predictions}.jsonl, grpo/
#   runs/gsm8k/Llama-3.2-1B-Instruct-px-mf/seed_S/full_grpo/{predictions.jsonl,grpo/}   (merged-fresh RL alone)
# Resumable: route_sft_grpo.py skips methods already finished in --output_dir and resumes GRPO from the last saved
# checkpoint, so a job that times out or fails is resubmitted with the identical command (re-run this script's line).
# One E2 job (2 methods) took 1:51-1:58 on an L40S; single-method jobs need about 1 h, the gold mf job about 2 h.
set -euo pipefail
cd /orcd/scratch/orcd/006/usemil/data_alloc
G=runs/gsm8k/Llama-3.2-1B-Instruct
mkdir -p logs
ENV='set -eo pipefail; module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; export PYTHONPATH=.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python -c "import torch; assert torch.cuda.is_available()"; nvidia-smi --query-gpu=name --format=csv,noheader;'
E2="python route_sft_grpo.py --model unsloth/Llama-3.2-1B-Instruct --dataset gsm8k --max_eval_samples -1 --bf16 --reward marked --reward_needs_stop --log_rollouts --panel_prefix '## Step' '## Solution' 'Solution:' --grpo_steps 150 --grpo_lr_scheduler linear --grpo_warmup_steps 10 --grpo_save_steps 25 --grpo_loss_type grpo --eval_temperature 0.6 --eval_k 4"
NORMAL="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_preemptable -t 24:00:00 --requeue --output=logs/%x-%j.out"   # preemptable: resumable runs; mit_normal_gpu is busy with prereg 16
TEACH="--sft_traces runs/gsm8k/teachers/qwen7b.jsonl"
sub() { local name=$1; shift; echo "$name: $(sbatch $NORMAL --job-name=$name --wrap "$ENV $*")"; }

for S in 42 43 44; do
  sub px-cont-gold-s$S    "$E2 --seed $S --reuse_from $G/seed_$S --methods random_sft_250 --output_dir $G-px-cont/seed_$S --wandb_tags prereg15,partX,cont"
  sub px-cont-teacher-s$S "$E2 --seed $S $TEACH --reuse_from $G-qwen7b/seed_$S --methods random_sft_250 --output_dir $G-px-cont-qwen7b/seed_$S --wandb_tags prereg15,partX,cont"
  sub px-mf-gold-s$S      "$E2 --seed $S --merge_sft_start --lora_init_seed $S --reuse_from $G/seed_$S --methods full_grpo random_sft_250 --output_dir $G-px-mf/seed_$S --wandb_tags prereg15,partX,mf"
  sub px-mf-teacher-s$S   "$E2 --seed $S --merge_sft_start --lora_init_seed $S $TEACH --reuse_from $G-qwen7b/seed_$S --methods random_sft_250 --output_dir $G-px-mf-qwen7b/seed_$S --wandb_tags prereg15,partX,mf"
done
# After all 12 jobs: python -I runs/theory/scratch/score_prereg15.py --part X --schema_only (via Slurm CPU), then score.
