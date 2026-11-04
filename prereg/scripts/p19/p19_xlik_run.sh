#!/usr/bin/env bash
# M2 (prereg 19) FULL cross-likelihood runs. NOT SUBMITTED by the tooling engineer: the lead submits after registration.
# One job per (family, seed) on mit_preemptable (requeue; xlik.py resumes per (scorer, source) block; a re-run block is
# appended again and the scorer keeps the last copy). Outputs: runs/probes/prereg19_xlik/<family>.jsonl (all seeds).
# Estimated GPU time on an L40S (smoke test, job 27655875): gsm8k ~0.6 h per seed; llama17b ~1.9 h per seed;
# gemma17b ~3.2 h per seed (H100/H200 nodes are expected to be faster; not measured).
#   bash runs/theory/scratch/p19_xlik_run.sh            # prints the sbatch commands (dry)
#   bash runs/theory/scratch/p19_xlik_run.sh --submit   # submits them, chained so that at most ONE GPU job runs at a time
set -euo pipefail
cd /orcd/scratch/orcd/006/usemil/data_alloc
ENV='set -eo pipefail; module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; cd /orcd/scratch/orcd/006/usemil/data_alloc; export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; nvidia-smi --query-gpu=name --format=csv,noheader;'
OPTS="--parsable --exclude=node2119,node3006 --gres=gpu:1 -c 8 --mem=64G -p mit_preemptable --requeue --output=runs/probes/prereg19_xlik/%x-%j.out"
mkdir -p runs/probes/prereg19_xlik
JOBS=()
for s in 42 43 44; do JOBS+=("gsm8k $s gold_250,teacher_250 02:00:00"); done
for s in 42 43; do JOBS+=("llama17b $s gold,teacher 04:00:00"); done
for s in 42 43; do JOBS+=("gemma17b $s gold,teacher 06:00:00"); done   # optional (Gemma = post hoc arms)
prev=""
for j in "${JOBS[@]}"; do
  read -r fam seed arms t <<< "$j"
  cmd="python -I runs/theory/scratch/xlik.py --family $fam --seed $seed --arms $arms --out runs/probes/prereg19_xlik/$fam.jsonl"
  dep=${prev:+--dependency=afterany:$prev}
  if [ "${1:-}" = "--submit" ]; then
    prev=$(sbatch $OPTS -t $t $dep --job-name=p19-xlik-$fam-s$seed --wrap "$ENV $cmd")
    echo "p19-xlik-$fam-s$seed: $prev"
  else
    echo "sbatch $OPTS -t $t $dep --job-name=p19-xlik-$fam-s$seed --wrap \"\$ENV $cmd\""
    prev=PREV
  fi
done
