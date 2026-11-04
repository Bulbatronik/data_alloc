#!/usr/bin/env bash
# Submit prereg/19 M6 (length bias in damaged starts): google/gemma-3-1b-it, MATH L3-5 (math35d), merged-fresh, gold and
# teacher arms, seeds 42-43, --grpo_loss_type dr_grpo (TRL: per-token loss summed and divided by a constant,
# B x max_completion_length, instead of each sequence's own length; reward std scaling unchanged, as in prereg-13 E2's
# dr_grpo runs). Run ONLY after prereg/19 is committed and pushed (refused otherwise unless FORCE=1); DRY=1 prints.
#
# Commands = the 17B Gemma gold / teacher merged-fresh lines (prereg/scripts/p17/submit_prereg17b.sh with PROTO=mf; gold as
# re-run by runs/theory/scratch/submit_prereg17b_gemma_goldfix.sh, i.e. with --sft_batch_size 2 --sft_grad_accum 4),
# changing ONLY:
#   --grpo_loss_type grpo  -> dr_grpo
#   + --reuse_from runs/math35d/gemma-p17b-mf-<arm>/seed_S   (default --reuse_stage sft: the 17B SFT adapter
#     <arm>/seed_S/listed_sft/sft is loaded instead of training SFT again; --merge_sft_start merges it, so the GRPO start
#     is the 17B start; --sft_ids / --sft_traces / --grpo_ids kept, so the SFT / RL split is 17B's)
#   --output_dir runs/math35d/gemma-p19dr-mf-<arm>/seed_S;  --wandb_tags prereg17,17B,... -> prereg19,M6,gemma,mf,<arm>,dr_grpo
# Unchanged: 300 steps, constant lr after 10 warm-up steps, checkpoints at 150 and 300, 1,024 tokens, batch 4 x 16,
# marked + stop reward, --log_rollouts, --max_eval_samples 100 (greedy in-pipeline check only), --lora_init_seed S.
# Infrastructure: mit_preemptable --requeue (as every Gemma job after 2026-10-24), -t 12:00:00, chain of afterany retries.
# WITH_RL=1 also runs RL alone (full_grpo) with dr_grpo, seeds 42-43 (not in MECH_PLAN M6; the registered truncation
# comparator is the 17B GRPO RL-alone runs; this would add a same-loss comparator, ~8 GPU-h).
#
# Evaluations as in 17B: probe.py --k 8, T 0.6, top-p 0.95, 1,024 tokens, all 1,000 test items, prompt boxed_user, at
# checkpoints 150 and 300 -> runs/probes/prereg19_dr/gemma_dr_s{150,300}.jsonl. The pre-RL start and the untrained model
# are the SAME models as 17B's, so their evaluations are reused: runs/probes/prereg17b/gemma_mf_{sft,untrained}.jsonl.
set -euo pipefail
cd /orcd/scratch/orcd/006/usemil/data_alloc
RETRY=${RETRY:-2}
ERETRY=${ERETRY:-3}
OUT=runs/probes/prereg19_dr
if [ -z "${FORCE:-}" ] && [ -z "${DRY:-}" ] && ! ls prereg/19_*.md > /dev/null 2>&1; then
  echo "prereg/19_*.md not found: register prereg 19 first (or FORCE=1)" >&2; exit 1
fi
for a in gold teacher; do for S in 42 43; do
  [ -s runs/math35d/gemma-p17b-mf-$a/seed_$S/listed_sft/sft/adapter_model.safetensors ] || { echo "missing 17B $a seed $S SFT adapter" >&2; exit 1; }
done; done
for f in runs/probes/prereg17b/gemma_mf_sft.jsonl runs/probes/prereg17b/gemma_mf_untrained.jsonl; do [ -s $f ] || { echo "missing reused $f" >&2; exit 1; }; done
mkdir -p logs $OUT
CENV='set -eo pipefail; module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; export PYTHONPATH=. HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1;'
ENV="$CENV"' export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python -c "import torch; assert torch.cuda.is_available()"; nvidia-smi --query-gpu=name --format=csv,noheader;'
GPU="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_preemptable -t 12:00:00 --requeue --output=logs/%x-%j.out"
# 17B COMMON, verbatim except --grpo_loss_type
COMMON="--dataset math35d --max_train_samples -1 --sft_max_length 2048 --prompt_style boxed_user --max_eval_samples 100 --bf16 --reward marked --reward_needs_stop --log_rollouts --grpo_steps 300 --grpo_lr_scheduler constant_with_warmup --grpo_warmup_steps 10 --grpo_save_steps 150 --grpo_batch_size 4 --grpo_grad_accum 16 --grpo_max_completion_length 1024 --eval_max_new_tokens 1024 --grpo_loss_type dr_grpo --grpo_ids data/math35/rl_pool_ids.json"
M=google/gemma-3-1b-it
FIX="--sft_batch_size 2 --sft_grad_accum 4"  # gold: the goldfix re-run's SFT batching (inert here: SFT is reused)

sb() { if [ -n "${DRY:-}" ]; then printf 'sbatch' >&2; printf ' %q' "$@" >&2; echo >&2; echo "DRY$RANDOM"; else sbatch "$@"; fi; }
chain() {  # chain NAME DEP N CMD -> N+1 jobs linked by afterany; echoes the last job id
  local name=$1 dep=$2 n=$3; shift 3
  local d=$dep j
  for ((i = 0; i <= n; i++)); do
    j=$(sb $GPU ${d:+--dependency=$d} --job-name=$name-r$i --wrap "$ENV $*")
    d="afterany:$j"
  done
  echo "$(date +%F_%T) $name: r0..r$n, last job $j" >&2
  echo "$j"
}

TRAIN=()
for S in 42 43; do
  MF="--merge_sft_start --lora_init_seed $S"
  for arm in gold teacher; do
    case $arm in gold) TR="gold $FIX" ;; teacher) TR=data/math35/traces/teacher_llama8b.jsonl ;; esac
    TRAIN+=($(chain p19dr-gemma-$arm-s$S "" $RETRY "python route_sft_grpo.py --model $M $COMMON --seed $S $MF --methods listed_sft --sft_ids data/math35/p17b_sft_ids_${arm}_gemma.json --sft_traces $TR --reuse_from runs/math35d/gemma-p17b-mf-$arm/seed_$S --output_dir runs/math35d/gemma-p19dr-mf-$arm/seed_$S --wandb_tags prereg19,M6,gemma,mf,$arm,dr_grpo"))
  done
  if [ -n "${WITH_RL:-}" ]; then
    TRAIN+=($(chain p19dr-gemma-grpo-s$S "" $RETRY "python route_sft_grpo.py --model $M $COMMON --seed $S $MF --methods full_grpo --output_dir runs/math35d/gemma-p19dr-mf-grpo/seed_$S --wandb_tags prereg19,M6,gemma,mf,grpo,dr_grpo"))
  fi
done
ALL="afterok:$(IFS=:; echo "${TRAIN[*]}")"
P="python probe.py --model $M --dataset math35d --prompt_style boxed_user --reward marked --temperature 0.6 --top_p 0.95 --max_new_tokens 1024 --batch_size 256"
for c in 150 300; do
  chain p19dr-ev-gemma-s$c "$ALL" $ERETRY "$P --k 8 --out $OUT/gemma_dr_s$c.jsonl --adapters 'runs/math35d/gemma-p19dr-mf-*/seed_*/*/grpo/checkpoint-$c'" > /dev/null
done
echo "$(date +%F_%T) submitted; after all chains: score_prereg19_kl.py --part M6 --schema_only, then --unblind" >&2
