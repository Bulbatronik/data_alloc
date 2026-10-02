# 02 — RL-budget controls (registered 2026-10-02, before any of these runs)

Common setup:
- Llama-3.2-1B-Instruct.
- Existing gold-trace SFT adapters, reused via `--reuse_from`.
- GRPO otherwise standard: G=8, 1 epoch, lr 2e-5 (1×), T=0.8.
- Full test set.
- GRPO seeds gs1 and gs2.
- "Recovered" = more than 50% of final greedy answers are `native` (`## Step`).
- Runs that diverge (NaN) are reported but not scored.

## Adapters (w0 measured earlier, `runs/gsm8k/Llama-3.2-1B-Instruct-sftscan/`)

| Name | SFT lr / seed / n | w0 | Outcome under standard GRPO (existing runs) |
|---|---|---|---|
| a016 | 2e-4 / 42 / 75 | 0.16% | 0/1 |
| a039 | 5e-5 / 42 / 250 | 0.39% | 0/2; at 2× lr: 0/2 |
| a068 | 1e-4 / 42 / 125 | 0.68% | not run |
| a105 | 1e-4 / 43 / 125 | 1.05% | 0/2; at 2× lr: 1/2 |
| a184 | 5e-5 / 42 / 200 | 1.84% | 0/1 |

## Model being tested

For softmax policy gradient on a style logit ℓ with fitness gap Δp, the expected drift is dℓ/dt ∝ η_t·w(1−w)·Δp. Starting from small w0, escape needs Σ_t η_t·c·Δp ≳ 1/w0. Hence:
- the threshold scales as w_crit ∝ 1/(Σ η_RL);
- rollouts per step do not move it.

**Calibration (only from existing 1-epoch data).** w_crit(1×) = 1.7% [1.1, 2.5] (`runs/theory/style_diagnostic.md`, Llama-1B w50). This gives:

| GRPO lr | w_crit | Interval |
|---|---|---|
| 2× | 0.85% | [0.55, 1.25] |
| 4× | 0.43% | [0.28, 0.63] |

Δp differs somewhat across adapters; this is a stated approximation.

## L — RL learning-rate ladder (16 runs)

Runs:
- 1× (2e-5): a068 gs1 (control).
- 2× (4e-5): a016, a068, a184, each gs1 and gs2.
- 4× (8e-5): a016, a039, a068, a105, a184, each gs1 and gs2.

**Decisive calls.** Recover iff w0 > w_crit; "borderline" = w0 inside the interval, not scored:

| Adapter | 2× | 4× |
|---|---|---|
| a016 | no | no |
| a039 | (existing) | borderline |
| a068 | borderline | **yes** |
| a105 | (existing) | **yes** |
| a184 | **yes** | **yes** |
| a068 at 1× | no | |

**Registered predictions:**
- **L1.** At least 10 of the 13 decisively-called new runs go as called: 2 + 2 at 2×, 2 + 2 + 2 + 2 at 4×, and 1 at 1×.
- **L2 (monotonicity).** For every adapter, the recovered fraction is non-decreasing in GRPO lr.

## P — top-p (8 runs)

Runs: GRPO at 1× with `--grpo_top_p 1.0` from a016, a039, a068 and a105, each gs1 and gs2.

**Hypotheses:**
- **H_full:** the threshold is set by the full share w0, as in the model above. Then top-p=1 changes little: a016, a039 and a068 stay (0/6), and a105 is borderline.
- **H_nucleus:** the threshold is set by the share that survives top-p 0.95. That share is 34–86% smaller in this range, so top-p=1 rescues a039 and a068.

**Registered prediction (H_full):**
- **P1.** At most 1 of the 4 runs from a039 and a068 recovers.

**Decision rule.** If 2 or more of those 4 recover, report that part of the "extinction" at top-p 0.95 is a nucleus-truncation effect, and restate the threshold in nucleus mass.

## M — matched budget for 2-epoch runs (3 runs)

Runs: GRPO alone (`full_grpo`), 2 epochs, seeds 42, 43, 44. Compare with the existing gold n=250 runs at 2 epochs:
- `trajectories/n250`: 51.5 / 50.3 / 51.1, mean 51.0.
- `extinction/extra_epoch` (+1 epoch): 52.8 / 52.2 / 51.6, mean 52.2.

**Registered prediction:**
- **M1.** Mean GRPO-alone (2 epochs) ≥ 54.2, i.e. at least 2.0 points above the higher of the two (52.2).

## Outcome (scored 2026-10-02 with `runs/theory/scratch/score_prereg02.py`)

All 27 runs completed; none diverged.

| Prediction | Result | Held? |
|---|---|---|
| L1 | 8/13 decisive calls correct (registered: at least 10) | **No** |
| L2 | recovered fraction never decreases with GRPO lr, for all 5 adapters | Yes |
| P1 | 0 of 4 runs from a039/a068 recovered; 0 of 8 for all top-p=1 runs | Yes |
| M1 | GRPO alone at 2 epochs: 55.3 / 56.2 / 54.7, mean 55.4 | Yes |

**Recovered / runs at 1× / 2× / 4× GRPO lr:**

| Adapter | w0 | 1× | 2× | 4× |
|---|---|---|---|---|
| a016 | 0.16% | 0/1 | 0/2 | 0/2 |
| a039 | 0.39% | 0/2 | 0/2 | 0/2 |
| a068 | 0.68% | 0/1 | 0/2 | 0/2 |
| a105 | 1.05% | 0/2 | 1/2 | 2/2 |
| a184 | 1.84% | 0/1 | 0/2 | 1/2 |

**L1 misses:**
- a184 at 2× (2 runs) and at 4× (1 run): predicted to recover, did not.
- a068 at 4× (2 runs): predicted to recover, did not.

**Reading:**
- A larger RL step helps, monotonically.
- A single w0 threshold scaled by 1/Ση does not predict *which* adapters recover.
- The stated approximation — the same Δp for all adapters — is the likely culprit, but that explanation is post hoc. One example: a184 (SFT lr 5e-5) reaches 50–57 while staying in the gold style, so its own style may have little fitness advantage.
- P1: at these shares, sampling without nucleus truncation does not rescue the style, so the failures are not an artefact of top-p 0.95.
- M1: at a matched 2-epoch budget, GRPO alone (55.4) beats gold n=250 (51.0 at 2 epochs; 52.2 with +1 epoch) by 3.2–4.4 points.
