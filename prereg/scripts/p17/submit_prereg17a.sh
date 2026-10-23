#!/usr/bin/env bash
# Submit prereg/17 Part 17A: GSM8K, Llama-3.2-1B-Instruct, the 08 S' protocol (continuing adapters, 375 GRPO steps,
# constant lr after 10 warm-up steps, marked reward, checkpoints every 75 steps), NEW arms only, seeds 42-44:
#   rft      SFT on the untrained model's own marked-correct samples (pool, k = 8, T 0.8) for its 250 SFT prompts
#   hs       head start: SFT on own marked-correct samples of the S' GRPO-alone checkpoint-75 (same seed), same prompts
#   tunc     teacher (Qwen2.5-7B-I) traces on pool prompts the untrained model never solves (lenient pass@8 = 0),
#            n <= 250 (p17_make_tunc_ids.py), RL on the rest of the 1,000-prompt pool (listed_sft)
#   goldlin  the S' gold n=250 adapters, GRPO with LINEAR decay over 375 steps (clock test)
#   grpolin  GRPO alone with the same linear schedule (goldlin's comparator)
# Reference arms (S' gold / teacher n=250 and GRPO alone) are reused; their checkpoints 150 / 300 are evaluated here.
# Run ONLY after prereg/17 is committed and pushed.
#
# Partition mit_normal_gpu, <= 6 h per job, --exclude=node2119,node3006. Every step is a CHAIN of RETRY+1 jobs linked by
# afterany: route_sft_grpo.py skips finished methods and resumes GRPO from the last checkpoint, and probe.py skips
# adapters already in its --out file, so a job killed at 6 h is continued by the next; a retry that finds the work done
# exits after the untrained greedy check (training) or immediately (probe). Downstream steps wait on afterok of the
# last job of each upstream chain.
# Estimated GPU time (prereg-14 throughputs: ~20k GSM8K samples / GPU-h; S' 375-step run ~1.5 h): traces 1.6 h;
# training 15 runs x ~1.8 h = 27 h; evaluations 93 adapter-evals x 5,276 samples = 0.49 M samples = 25 h. Total ~53 GPU-h
# (~27 h wall-clock at the 2-GPU limit).
set -euo pipefail
cd /orcd/scratch/orcd/006/usemil/data_alloc
G=runs/gsm8k/Llama-3.2-1B-Instruct
TR=$G/traces
OUT=runs/probes/prereg17a
RETRY=${RETRY:-2}
mkdir -p logs $OUT
CENV='set -eo pipefail; module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; export PYTHONPATH=.;'
ENV="$CENV"' export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python -c "import torch; assert torch.cuda.is_available()"; nvidia-smi --query-gpu=name --format=csv,noheader;'
GPU="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_preemptable -t 06:00:00 --requeue --output=logs/%x-%j.out"
CPU="--parsable --exclude=node2119,node3006 -c 4 --mem=16G -p mit_normal -t 00:30:00 --output=logs/%x-%j.out"

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

PROBE="python probe.py --model unsloth/Llama-3.2-1B-Instruct --dataset gsm8k --reward marked --top_p 0.95 --max_new_tokens 512 --batch_size 128"
# 08 S' training command (W&B run l6fbde5d) + --log_rollouts (measurement only, for the stop-rate exclusion)
S1="python route_sft_grpo.py --model unsloth/Llama-3.2-1B-Instruct --dataset gsm8k --max_eval_samples -1 --bf16 --reward marked --grpo_steps 375 --grpo_warmup_steps 10 --grpo_save_steps 75 --log_rollouts"
CONST="--grpo_lr_scheduler constant_with_warmup"
LIN="--grpo_lr_scheduler linear"

# 1. SFT traces (k = 8 at the RL sampling temperature; probe grades with the marked grader, so `correct` = marked)
T_SELF=$(chain p17a-tr-self "" "$PROBE --on pool --k 8 --temperature 0.8 --out $TR/p17_self_k8.jsonl")
declare -A T_HS
for S in 42 43 44; do
  T_HS[$S]=$(chain p17a-tr-hs-s$S "" "$PROBE --on pool --k 8 --temperature 0.8 --out $TR/p17_hs_s$S.jsonl --adapters $G-marked-steps375/seed_$S/full_grpo/grpo/checkpoint-75")
done
IDS=$(sbatch $CPU --dependency=afterok:$T_SELF --job-name=p17a-tunc-ids --wrap "$CENV python -I runs/theory/scratch/p17_make_tunc_ids.py")

# 2. training
TRAIN=()
for S in 42 43 44; do
  TRAIN+=($(chain p17a-rft-s$S "afterok:$T_SELF" "$S1 $CONST --seed $S --sft_traces $TR/p17_self_k8.jsonl --methods random_sft_250 --output_dir $G-p17a-rft/seed_$S --wandb_tags prereg17,17A,rft"))
  TRAIN+=($(chain p17a-hs-s$S "afterok:${T_HS[$S]}" "$S1 $CONST --seed $S --sft_traces $TR/p17_hs_s$S.jsonl --methods random_sft_250 --output_dir $G-p17a-hs/seed_$S --wandb_tags prereg17,17A,hs"))
  TRAIN+=($(chain p17a-tunc-s$S "afterok:$IDS" "$S1 $CONST --seed $S --methods listed_sft --sft_ids $TR/p17_tunc_ids.json --sft_traces runs/gsm8k/teachers/qwen7b.jsonl --output_dir $G-p17a-tunc/seed_$S --wandb_tags prereg17,17A,tunc"))
  TRAIN+=($(chain p17a-goldlin-s$S "" "$S1 $LIN --seed $S --reuse_from $G/seed_$S --methods random_sft_250 --output_dir $G-p17a-goldlin/seed_$S --wandb_tags prereg17,17A,goldlin"))
  TRAIN+=($(chain p17a-grpolin-s$S "" "$S1 $LIN --seed $S --methods full_grpo --output_dir $G-p17a-grpolin/seed_$S --wandb_tags prereg17,17A,grpolin"))
done
ALL="afterok:$(IFS=:; echo "${TRAIN[*]}")"

# 3. evaluations (T 0.6, k = 4, all 1,319 test items; probe.py)
E="$PROBE --k 4 --temperature 0.6"
chain p17a-ev-ref "" "$E --out $OUT/ref_s150.jsonl --adapters '$G-marked-steps375/seed_*/full_grpo/grpo/checkpoint-150' '$G-marked-steps375/seed_*/random_sft_250/grpo/checkpoint-150' && $E --out $OUT/ref_s300.jsonl --adapters '$G-marked-steps375/seed_*/full_grpo/grpo/checkpoint-300'" > /dev/null
chain p17a-ev-sft "$ALL" "$E --out $OUT/new_sft.jsonl --adapters '$G-p17a-rft/seed_*/random_sft_250/sft' '$G-p17a-hs/seed_*/random_sft_250/sft' '$G-p17a-tunc/seed_*/listed_sft/sft'" > /dev/null
for c in 75 150 225 300 375; do
  chain p17a-ev-s$c "$ALL" "$E --out $OUT/new_s$c.jsonl --adapters '$G-p17a-*/seed_*/*/grpo/checkpoint-$c'" > /dev/null
done
echo "submitted; after all chains: sbatch runs/theory/scratch/run_p17_dry.sh --part 17A --schema_only, then score with --unblind"
