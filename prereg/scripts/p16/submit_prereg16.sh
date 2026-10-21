#!/usr/bin/env bash
# Submit prereg/16 (OOD retention): 40 adapters x {GSM-Plus, SVAMP, MATH-600}, sampled T 0.6, marked reward field.
# probe.py skips adapters already present in --out, so any job can be resubmitted after a timeout.
# Run ONLY after prereg/16 is committed and pushed.
# Sizing (14's measured throughputs: ~20k GSM-like and ~7.9k MATH samples per GPU-h):
#   GSM-Plus k=2 and SVAMP k=4: <= 21 adapters per job, about 4-4.5 h.
#   MATH (600 x k=4 = 2.4k samples per adapter): split by source, <= 12 adapters per job, <= 3.6 h.
set -euo pipefail
cd /orcd/scratch/orcd/006/usemil/data_alloc
G=runs/gsm8k/Llama-3.2-1B-Instruct
OUT=runs/probes/prereg16
mkdir -p $OUT logs
ENV='set -eo pipefail; module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; export PYTHONPATH=.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python -c "import torch; assert torch.cuda.is_available()"; nvidia-smi --query-gpu=name --format=csv,noheader;'
PROBE="python probe.py --model unsloth/Llama-3.2-1B-Instruct --reward marked --temperature 0.6 --top_p 0.95 --batch_size 128"
NORMAL="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_normal_gpu -t 06:00:00 --output=logs/%x-%j.out"
SFT_G="'$G/seed_*/random_sft_50/sft' '$G/seed_*/random_sft_250/sft' '$G/seed_*/random_sft_500/sft'"
SFT_T="'$G-qwen7b/seed_*/random_sft_50/sft' '$G-qwen7b/seed_*/random_sft_250/sft' '$G-qwen7b/seed_*/random_sft_500/sft'"
RL_G="'$G-marked-steps375/seed_*/*/grpo/checkpoint-375'"          # full_grpo + gold 50/250/500 (12)
RL_T="'$G-marked-steps375-qwen7b/seed_*/*/grpo/checkpoint-375'"   # teacher 50/250/500 (9)
sub() { local name=$1; shift; echo "$name: $(sbatch $NORMAL --job-name=$name --wrap "$ENV $*")"; }

for ds in gsm_plus svamp; do
  case $ds in
    gsm_plus) A="$PROBE --dataset gsm_plus --k 2 --max_new_tokens 512" ;;
    svamp)    A="$PROBE --dataset svamp --k 4 --max_new_tokens 512" ;;
  esac
  sub p16-$ds-sft "${A/--k [0-9]/--k 8} --out $OUT/${ds}_untrained.jsonl && $A --out $OUT/${ds}_sft.jsonl --adapters $SFT_G $SFT_T"
  sub p16-$ds-rl  "$A --out $OUT/${ds}_s375.jsonl --adapters $RL_G $RL_T"
done
A="$PROBE --dataset math --n_prompts 600 --k 4 --max_new_tokens 1024"
sub p16-math-sftg "${A/--k 4/--k 8} --out $OUT/math_untrained.jsonl && $A --out $OUT/math_sft.jsonl --adapters $SFT_G"
sub p16-math-sftt "$A --out $OUT/math_sft_t.jsonl --adapters $SFT_T"
sub p16-math-rlg  "$A --out $OUT/math_s375.jsonl --adapters $RL_G"
sub p16-math-rlt  "$A --out $OUT/math_s375_t.jsonl --adapters $RL_T"
# MATH outputs are split by source (math_sft + math_sft_t, math_s375 + math_s375_t); the scorer concatenates them.
