#!/usr/bin/env bash
# Submit prereg/17 Part 17B: MATH L3-5 (math35d: fixed SFT pool 1,500, RL pool 1,000, test 1,000 = 367 MATH-500 L3-5
# + 633 other), two students (Llama-3.2-1B-Instruct, Gemma-3-1B-it), arms RL alone (seeds 42-44), gold (Hendrycks
# solutions), teacher (Llama-3.1-8B-Instruct traces), RFT (own marked-correct samples) (seeds 42-43), n = 500 SFT items
# per arm (p17_make_math_ids.py), GRPO 300 steps, 1,024 tokens, marked + stop reward, constant lr after 10 warm-up
# steps, checkpoints at 150 and 300. Evaluations (probe.py, T 0.6, top-p 0.95, 1,024 tokens, all 1,000 test items):
# untrained k = 24; after SFT, checkpoint-150 and checkpoint-300 k = 8.
# Run ONLY after prereg/17 is committed and pushed, and with PROTO set by the lead after Part X:
#   PROTO=mf   merged SFT + fresh LoRA (--merge_sft_start --lora_init_seed S, RL alone included)   [if X1 holds]
#   PROTO=cont continuing adapter (RL alone: fresh LoRA drawn by TRL)                              [otherwise]
# Prerequisites (checked below): the dataset engineer's trace files data/math35/traces/{teacher_llama8b,llama1b,
# gemma1b}.jsonl, and a route_sft_grpo.py option --grpo_ids that restricts the GRPO bucket to the RL pool. Without it,
# math35d's training split (SFT pool + RL pool) would put the 1,000 unused SFT-pool items into every RL bucket and RL
# alone would train on all 2,500 items. It does not exist as of 2026-10-23.
#
# Partition mit_normal_gpu, <= 6 h per job, --exclude=node2119,node3006; every step is a chain of afterany retries
# (training resumes from the last checkpoint; probe skips finished adapters).
# Estimated GPU time (prereg-14 Part C scaled: ~3.9 h per 375 steps at 1,024 tokens on MATH L1-3 x 0.8 for 300 steps
# x 1.3 for L3-5 lengths; Gemma x 1.3, unverified; probe ~6k samples / GPU-h): training 18 runs ~81 h; evaluations
# 2 x 216k samples ~72 h; total ~155 GPU-h (~3.3 days wall-clock at the 2-GPU limit). Replace with the dataset
# engineer's measured timing (S3 smoke job 26526538) when available.
set -euo pipefail
cd /orcd/scratch/orcd/006/usemil/data_alloc
PROTO=${PROTO:?set PROTO=mf or PROTO=cont (the lead decides after Part X)}
RETRY=${RETRY:-2}
ERETRY=${ERETRY:-4}
OUT=runs/probes/prereg17b
mkdir -p logs $OUT
grep -q -- '"--grpo_ids"' route_sft_grpo.py || { echo "route_sft_grpo.py has no --grpo_ids: 17B cannot restrict GRPO to the RL pool" >&2; exit 1; }
for f in teacher_llama8b llama1b gemma1b; do [ -s data/math35/traces/$f.jsonl ] || { echo "missing data/math35/traces/$f.jsonl" >&2; exit 1; }; done
CENV='set -eo pipefail; module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; export PYTHONPATH=. HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1;'
ENV="$CENV"' export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python -c "import torch; assert torch.cuda.is_available()"; nvidia-smi --query-gpu=name --format=csv,noheader;'
GPU="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_normal_gpu -t 06:00:00 --output=logs/%x-%j.out"
CPU="--parsable --exclude=node2119,node3006 -c 4 --mem=16G -p mit_normal -t 00:30:00 --output=logs/%x-%j.out"

chain() {  # chain NAME DEP N CMD -> N+1 jobs linked by afterany; echoes the last job id
  local name=$1 dep=$2 n=$3; shift 3
  local d=$dep j
  for ((i = 0; i <= n; i++)); do
    j=$(sbatch $GPU ${d:+--dependency=$d} --job-name=$name-r$i --wrap "$ENV $*")
    d="afterany:$j"
  done
  echo "$name: last job $j" >&2
  echo "$j"
}

IDS=$(sbatch $CPU --job-name=p17b-ids --wrap "$CENV python -I runs/theory/scratch/p17_make_math_ids.py")
declare -A MODEL=([llama]=unsloth/Llama-3.2-1B-Instruct [gemma]=google/gemma-3-1b-it)
declare -A OWN=([llama]=llama1b [gemma]=gemma1b)
COMMON="--dataset math35d --max_train_samples -1 --sft_max_length 2048 --prompt_style boxed_user --max_eval_samples 100 --bf16 --reward marked --reward_needs_stop --log_rollouts --grpo_steps 300 --grpo_lr_scheduler constant_with_warmup --grpo_warmup_steps 10 --grpo_save_steps 150 --grpo_batch_size 4 --grpo_grad_accum 16 --grpo_max_completion_length 1024 --eval_max_new_tokens 1024 --grpo_loss_type grpo --grpo_ids data/math35/rl_pool_ids.json"
for fam in llama gemma; do
  M=${MODEL[$fam]}
  R="runs/math35d/$fam-p17b-$PROTO"
  TRAIN=()
  for S in 42 43 44; do
    MF=$([ "$PROTO" = mf ] && echo "--merge_sft_start --lora_init_seed $S" || true)
    TRAIN+=($(chain p17b-$fam-$PROTO-grpo-s$S "afterok:$IDS" $RETRY "python route_sft_grpo.py --model $M $COMMON --seed $S $MF --methods full_grpo --output_dir $R-grpo/seed_$S --wandb_tags prereg17,17B,$fam,$PROTO,grpo"))
    [ $S = 44 ] && continue
    for arm in gold teacher rft; do
      case $arm in gold) TR=gold ;; teacher) TR=data/math35/traces/teacher_llama8b.jsonl ;; rft) TR=data/math35/traces/${OWN[$fam]}.jsonl ;; esac
      TRAIN+=($(chain p17b-$fam-$PROTO-$arm-s$S "afterok:$IDS" $RETRY "python route_sft_grpo.py --model $M $COMMON --seed $S $MF --methods listed_sft --sft_ids data/math35/p17b_sft_ids_${arm}_$fam.json --sft_traces $TR --output_dir $R-$arm/seed_$S --wandb_tags prereg17,17B,$fam,$PROTO,$arm"))
    done
  done
  ALL="afterok:$(IFS=:; echo "${TRAIN[*]}")"
  P="python probe.py --model $M --dataset math35d --prompt_style boxed_user --reward marked --temperature 0.6 --top_p 0.95 --max_new_tokens 1024 --batch_size 256"
  chain p17b-ev-$fam-$PROTO-untrained "" $ERETRY "$P --k 24 --out $OUT/${fam}_${PROTO}_untrained.jsonl" > /dev/null
  chain p17b-ev-$fam-$PROTO-sft "$ALL" $ERETRY "$P --k 8 --out $OUT/${fam}_${PROTO}_sft.jsonl --adapters '$R-*/seed_*/listed_sft/sft'" > /dev/null
  for c in 150 300; do
    chain p17b-ev-$fam-$PROTO-s$c "$ALL" $ERETRY "$P --k 8 --out $OUT/${fam}_${PROTO}_s$c.jsonl --adapters '$R-*/seed_*/*/grpo/checkpoint-$c'" > /dev/null
  done
done
echo "submitted; after all chains: run_p17_dry.sh --part 17B --schema_only --proto $PROTO, then score with --unblind"
