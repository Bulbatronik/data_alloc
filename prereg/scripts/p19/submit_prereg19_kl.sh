#!/usr/bin/env bash
# Submit prereg/19 M5 (KL anchoring is the switch): GSM8K, unsloth/Llama-3.2-1B-Instruct, merged-fresh protocol for EVERY
# arm, so that TRL's KL reference is each arm's own start (TRL 1.14 + PEFT 0.21: with --grpo_kl_beta > 0 and a PEFT model,
# GRPOTrainer adds a frozen copy of the starting adapter, "ref"; under --merge_sft_start that copy has B = 0, so the
# reference is exactly the merged SFT start, or the untrained model for RL alone; verified in the smoke test, report
# runs/reviews/round4/scratch/T19_kl_tooling_report.md).
# Arms: gold-250, teacher-250 (qwen7b traces) and RL alone; beta in {0, 0.02, 0.1, 0.5}; seeds 42-44; 300 GRPO steps:
# 36 GRPO runs in 24 training chains (the gold job also runs full_grpo, as Part X's px-mf-gold job did).
# Run ONLY after prereg/19 is committed and pushed (the lead registers and launches; refused otherwise unless FORCE=1).
# DRY=1 prints every sbatch line instead of submitting.
#
# Run commands = the registered Part X merged-fresh lines (prereg/scripts/p15/submit_prereg15x.sh, px-mf-gold-s$S and
# px-mf-teacher-s$S), changing ONLY:
#   --grpo_steps 150                 -> 300
#   --grpo_lr_scheduler linear       -> constant_with_warmup   (--grpo_warmup_steps 10 kept: constant lr after 10 warm-up steps)
#   --grpo_save_steps 25             -> 100                    (checkpoints 100 / 200 / 300; a requeued run resumes from them)
#   (default --grpo_kl_beta 0)       -> --grpo_kl_beta $B
#   --output_dir ...-px-mf[-qwen7b]  -> ...-p19kl-b$B-mf[-qwen7b];  --wandb_tags prereg15,partX,mf -> prereg19,M5,mf,b$B
# Unchanged from Part X (E2's command line without --censor): --max_eval_samples -1, --bf16, --reward marked
# --reward_needs_stop, --log_rollouts, the three panel prefixes, --grpo_loss_type grpo, --eval_temperature 0.6 --eval_k 4
# (the in-pipeline untrained / after-SFT (merged start) / final evaluations; the final one is the scorer's step-300
# replication set), --merge_sft_start --lora_init_seed $S, --reuse_from the 08 S' SFT adapters, teacher --sft_traces,
# GRPO defaults (8 prompts x 8 samples, T 0.8, top-p 0.95, 512 tokens, lr 2e-5), mit_preemptable --requeue -t 24:00:00.
# Infrastructure only: HF_*_OFFLINE=1 (as 17B; models and data are cached) and the chain-with-retries pattern (17B):
# every training line is a chain of 1 + RETRY jobs linked by afterany; route_sft_grpo skips finished methods and resumes
# GRPO from the last checkpoint, so a retry after a finished run exits in seconds. (beta > 0 resumes need the
# load_resume_default_adapter fix in route_sft_grpo.py; without it a resumed run silently restarts from the step-0 LoRA.)
#
# Evaluations (probe.py with prereg-14 Part A's settings: --reward marked --temperature 0.6 --top_p 0.95 --batch_size 128
# --dataset gsm8k --k 4 --max_new_tokens 512, all 1,319 test items), into runs/probes/prereg19_kl/:
#   start.jsonl        the 6 merged SFT starts (probe --merged_start on the gold/teacher random_sft_250/sft adapters, seeds
#                      42-44); independent of training, so it starts at once. KL of each start from the base (context):
#                      kl_start_vs_base.jsonl (kl_travel on those samples; pi_t = SFT adapter, pi_0 = base).
#   untrained          REUSED: runs/probes/prereg14_a/untrained.jsonl (same command and settings; sampling code unchanged
#                      since 2026-10-12) + E2's base_predictions (12 samples), the prereg-15 convention.
#   b<B>_s<t>.jsonl    the 9 checkpoints (gold, teacher, RL alone x 3 seeds) of beta B at t = 100, 200, 300, then in the
#                      same job kl_b<B>_s<t>.jsonl = kl_travel.py on those samples (per-sample KL(pi_t || pi_0), pi_0 = the
#                      run's own start; kl@0.6 primary, kl@1 reported). 12 chains, each after its beta's 6 training chains.
# probe.py and kl_travel.py skip adapters already in --out, so eval retries are cheap.
set -euo pipefail
cd /orcd/scratch/orcd/006/usemil/data_alloc
RETRY=${RETRY:-2}
ERETRY=${ERETRY:-2}
BETAS=${BETAS:-"0 0.02 0.1 0.5"}
OUT=runs/probes/prereg19_kl
G=runs/gsm8k/Llama-3.2-1B-Instruct
if [ -z "${FORCE:-}" ] && [ -z "${DRY:-}" ] && ! ls prereg/19_*.md > /dev/null 2>&1; then
  echo "prereg/19_*.md not found: register prereg 19 first (or FORCE=1)" >&2; exit 1
fi
grep -q 'def load_resume_default_adapter' route_sft_grpo.py || { echo "route_sft_grpo.py lacks the beta > 0 resume fix" >&2; exit 1; }
grep -q '"--merged_start"' probe.py || { echo "probe.py lacks --merged_start" >&2; exit 1; }
[ -s runs/probes/prereg14_a/untrained.jsonl ] || { echo "missing reused untrained evaluation" >&2; exit 1; }
for S in 42 43 44; do for d in $G $G-qwen7b; do [ -s $d/seed_$S/random_sft_250/sft/adapter_model.safetensors ] || { echo "missing $d/seed_$S SFT adapter" >&2; exit 1; }; done; done
mkdir -p logs $OUT
ENV='set -eo pipefail; module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; export PYTHONPATH=. HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python -c "import torch; assert torch.cuda.is_available()"; nvidia-smi --query-gpu=name --format=csv,noheader;'
TRAIN_GPU="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_preemptable -t 24:00:00 --requeue --output=logs/%x-%j.out"
EVAL_GPU="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_preemptable -t 12:00:00 --requeue --output=logs/%x-%j.out"
M5="python route_sft_grpo.py --model unsloth/Llama-3.2-1B-Instruct --dataset gsm8k --max_eval_samples -1 --bf16 --reward marked --reward_needs_stop --log_rollouts --panel_prefix '## Step' '## Solution' 'Solution:' --grpo_steps 300 --grpo_lr_scheduler constant_with_warmup --grpo_warmup_steps 10 --grpo_save_steps 100 --grpo_loss_type grpo --eval_temperature 0.6 --eval_k 4"
TEACH="--sft_traces runs/gsm8k/teachers/qwen7b.jsonl"
PROBE="python probe.py --model unsloth/Llama-3.2-1B-Instruct --reward marked --temperature 0.6 --top_p 0.95 --batch_size 128 --dataset gsm8k --k 4 --max_new_tokens 512"
KLT="python runs/theory/scratch/kl_travel.py --model unsloth/Llama-3.2-1B-Instruct --dataset gsm8k"

sb() {  # sbatch, or print it (DRY=1) and return a fake job id
  if [ -n "${DRY:-}" ]; then printf 'sbatch' >&2; printf ' %q' "$@" >&2; echo >&2; echo "DRY$RANDOM"; else sbatch "$@"; fi
}
chain() {  # chain NAME DEP N OPTS CMD -> N+1 jobs linked by afterany; echoes the last job id
  local name=$1 dep=$2 n=$3 opts=$4; shift 4
  local d=$dep j
  for ((i = 0; i <= n; i++)); do
    j=$(sb $opts ${d:+--dependency=$d} --job-name=$name-r$i --wrap "$ENV $*")
    d="afterany:$j"
  done
  echo "$(date +%F_%T) $name: r0..r$n, last job $j" >&2
  echo "$j"
}

chain p19kl-ev-start "" $ERETRY "$EVAL_GPU" "$PROBE --merged_start --out $OUT/start.jsonl --adapters $(for S in 42 43 44; do echo -n "'$G/seed_$S/random_sft_250/sft' '$G-qwen7b/seed_$S/random_sft_250/sft' "; done) && $KLT --samples $OUT/start.jsonl --out $OUT/kl_start_vs_base.jsonl" > /dev/null
for B in $BETAS; do
  TRAIN=()
  for S in 42 43 44; do
    TRAIN+=($(chain p19kl-b$B-gold-s$S "" $RETRY "$TRAIN_GPU" "$M5 --grpo_kl_beta $B --seed $S --merge_sft_start --lora_init_seed $S --reuse_from $G/seed_$S --methods full_grpo random_sft_250 --output_dir $G-p19kl-b$B-mf/seed_$S --wandb_tags prereg19,M5,mf,b$B"))
    TRAIN+=($(chain p19kl-b$B-teacher-s$S "" $RETRY "$TRAIN_GPU" "$M5 --grpo_kl_beta $B --seed $S --merge_sft_start --lora_init_seed $S $TEACH --reuse_from $G-qwen7b/seed_$S --methods random_sft_250 --output_dir $G-p19kl-b$B-mf-qwen7b/seed_$S --wandb_tags prereg19,M5,mf,b$B"))
  done
  ALL="afterok:$(IFS=:; echo "${TRAIN[*]}")"
  for t in 100 200 300; do
    chain p19kl-ev-b$B-s$t "$ALL" $ERETRY "$EVAL_GPU" "$PROBE --out $OUT/b${B}_s$t.jsonl --adapters '$G-p19kl-b$B-mf/seed_*/*/grpo/checkpoint-$t' '$G-p19kl-b$B-mf-qwen7b/seed_*/*/grpo/checkpoint-$t' && $KLT --samples $OUT/b${B}_s$t.jsonl --out $OUT/kl_b${B}_s$t.jsonl" > /dev/null
  done
done
echo "$(date +%F_%T) submitted; after all chains: score_prereg19_kl.py --part M5 --schema_only, then --unblind (CPU via Slurm)" >&2
