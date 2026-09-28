# SFT vs GRPO allocation curve

This project tests the data-allocation question directly:

> Given a fixed pool of `N` GSM8K training problems, how many should be used as SFT demonstrations and how many should be used as GRPO prompts?

For every point on the curve:

```text
num_grpo = N - num_sft
```

You can specify either fractions (`--fractions`) or exact x-axis counts (`--counts`).

The final plot is:

```text
x-axis: exact number of SFT examples
         (top axis shows N - num_sft GRPO examples)
y-axis: final GSM8K exact-match accuracy
```

Two allocation rules are compared at the same budget:

- **random**: randomly choose exactly `num_sft` examples for SFT. Within a seed, the same random permutation is reused at every x-value, so the random SFT buckets are nested as the budget grows.
- **adaptive**: choose exactly `num_sft` examples with the lowest GRPO-preference score for SFT; the rest go to GRPO.

The two endpoints are shared to avoid redundant training:

- `num_sft = 0`: full GRPO.
- `num_sft = N`: full SFT.

For every mixed run the training order is:

```text
same base model
   -> SFT on the SFT bucket
   -> GRPO on the disjoint GRPO bucket
   -> greedy GSM8K evaluation
```

## Adaptive routing rule

The frozen base policy is probed before training. For example `i`:

```text
p_i = fraction of K probe generations with the correct final answer

GRPO_value_i = 1 - p_i^G - (1-p_i)^G
SFT_value_i  = (1-p_i) * normalized_gold_trace_NLL_i

GRPO_preference_i = GRPO_value_i - beta * SFT_value_i
```

At a fixed SFT budget, examples are sorted by `GRPO_preference`. The lowest-scoring examples become SFT data and the highest-scoring examples become GRPO data.

This gives a budget-matched test of the routing idea rather than allowing the adaptive method to simply choose a different amount of SFT.

## Install

```bash
srun -p mit_normal -c 4 --mem=16G -t 01:00:00 bash setup_env.sh   # once
wandb login                                                        # once, if not already
```

W&B: project `sft-grpo-routing`, one group per model (e.g. `Qwen2.5-0.5B`). To run without W&B, pass `--wandb_mode disabled`.

## Run on the cluster (Slurm)

```bash
./run_all.sh                     # one job per seed on mit_preemptable (1 GPU each)
# ...after all jobs finish:
srun -p mit_normal -c 2 --mem=8G -t 00:30:00 ./run_all.sh aggregate
```

Outputs go to `runs/<model name>/seed_<k>/<method>/`: `sft/` and `grpo/` hold the LoRA adapter after each stage, plus `predictions.jsonl`, `predictions_after_sft.jsonl` (mixed runs) and `partition.json`. The untrained model is evaluated once per seed (`base_eval.json`); `results.csv` has `base_accuracy`, `after_sft_accuracy`, `sft_gain` and `grpo_gain`. Jobs are requeued on preemption and skip finished allocations.

## Run the original 0/25/50/75/100% comparison

One seed:

```bash
python run_curve.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max_train_samples 1000 \
  --max_eval_samples 300 \
  --fractions 0 0.25 0.50 0.75 1.0 \
  --seeds 42 \
  --strategies random adaptive \
  --bf16 \
  --gradient_checkpointing
```

For a proper comparison, use at least three seeds:

```bash
python run_curve.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max_train_samples 1000 \
  --max_eval_samples 300 \
  --fractions 0 0.25 0.50 0.75 1.0 \
  --seeds 42 43 44 \
  --strategies random adaptive \
  --bf16 \
  --gradient_checkpointing
```

Or simply:

```bash
./run_all.sh
```

## Run a smoother allocation curve

```bash
python run_curve.py \
  --max_train_samples 1000 \
  --max_eval_samples 300 \
  --fractions 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --seeds 42 43 44 \
  --strategies random adaptive \
  --bf16 \
  --gradient_checkpointing
```

With `N=1000`, the x-axis is therefore `0,100,...,1000`, and the corresponding GRPO counts are `1000,900,...,0`.

## W&B organization

Each trained allocation is one W&B run. Its config/summary includes:

- `num_sft`
- `num_grpo`
- `sft_fraction`
- routing `strategy`
- training seed and fixed dataset seed
- SFT / GRPO hyperparameters
- `final/accuracy`

All runs from one invocation are placed into one W&B **group**. At the end, `run_curve.py` creates a `curve-summary` run containing:

- a raw-results W&B Table,
- an aggregated mean/std Table,
- a W&B multi-line plot of `accuracy_mean` vs `n_sft`, grouped by routing strategy,
- the local publication-style PNG plot.

TRL's normal SFT/GRPO trainer metrics are also sent to the individual W&B runs unless you pass `--no-wandb_log_train`.

## Output files

The top-level curve directory contains:

```text
sweep_plan.json
curve_raw.csv
curve_summary.csv
allocation_curve.png
allocation_curve.pdf
seed_42/
seed_43/
seed_44/
```

Each seed directory additionally contains partitions, per-method predictions and router features.

`curve_summary.csv` has one row per `(strategy, num_sft)` with:

```text
accuracy_mean
accuracy_std
accuracy_min
accuracy_max
n_seeds
```

## Arbitrary exact allocation counts

`route_sft_grpo.py` now accepts method names with exact counts. For a 1000-example pool:

```bash
python route_sft_grpo.py \
  --max_train_samples 1000 \
  --methods \
      full_grpo \
      random_sft_250 adaptive_sft_250 \
      random_sft_500 adaptive_sft_500 \
      random_sft_750 adaptive_sft_750 \
      full_sft \
  --bf16
```

The old `random_25_75` / `adaptive_25_75` names remain supported.

## Experimental detail

`--data_seed` is separate from `--seed`. The dataset subset is therefore held fixed across seeds, while partition choice, model sampling, and training can vary. This prevents a change in the underlying GSM8K subset from being confounded with the allocation curve.

The experiment fixes **number of training problems**, not total FLOPs. SFT and GRPO have different compute costs, so this answers a data-allocation question. A compute-matched experiment would be a separate ablation.

## Exact-count example

If you want the x-axis to be specific sample counts rather than fractions:

```bash
python run_curve.py \
  --max_train_samples 1000 \
  --counts 0 100 250 500 750 900 1000 \
  --seeds 42 43 44 \
  --strategies random adaptive \
  --bf16
```
