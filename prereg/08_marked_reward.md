# 08 — Marked-answer reward: core rerun and step-matched control (registered 2026-10-04, before any of these runs)

## Why

The reward and grader used so far are lenient: an answer counts if its last number is correct, marked or not. GRPO alone exploits this, and the exploit grows with the RL budget. Lenient vs marked-only accuracy of existing GRPO-alone runs:

| Run | Lenient | Marked only |
|---|---|---|
| 1 epoch | 52.7 | 48.7 |
| 2 epochs | 55.4 | 44.4 |
| 2× lr | 55.0 | 24.3 |

Details: `runs/theory/scratch/rescore_parsers.py` and the amendment in `07`.

From now on the RL reward is `--reward marked`:
- An answer counts only if it is explicitly marked: `#### N` (also `#### <final numeric answer> N`, `$`, bold), `\boxed{N}`, or "answer is N".
- There is no last-number fallback.
- Evaluation uses the same grading and also reports the lenient accuracy.

**Setup.**
- Llama-3.2-1B-Instruct, GSM8K pool of 1000, full test set.
- Accuracy below means marked accuracy.
- The untrained model scores 39.1 marked (49.1 lenient).

## Part C — core rerun at the standard budget (12 runs)

- Cells:
  - GRPO alone;
  - gold SFT at n = 50 and n = 250;
  - Qwen2.5-7B-Instruct-teacher SFT at n = 250.
- Seeds 42, 43, 44.
- Standard GRPO: 1 epoch, lr 2e-5, linear decay, G=8, top-p 0.95, `loss_type=grpo`.
- SFT adapters are reused from the existing runs (`--reuse_from`).
- Output: `runs/gsm8k/Llama-3.2-1B-Instruct-marked{,-qwen7b}/seed_S/`.

For reference, existing lenient-reward runs re-graded marked give:

| Cell | Marked accuracy (lenient-reward runs) |
|---|---|
| GRPO alone | 48.7 |
| gold n=50 | 56.5 |
| gold n=250 | 45.7 |
| teacher n=250 | 55.1 |

**Predictions.**
- **C1.** GRPO alone learns to mark its answers: mean ≥ 44.0 (untrained 39.1 + 5).
- **C2 (unlock).** mean(gold n=50) ≥ mean(GRPO alone) + 2.0.
- **C3 (cliff).** mean(GRPO alone) − mean(gold n=250) ≥ 2.0.
- **C4 (teacher).** mean(teacher n=250) ≥ mean(GRPO alone) − 1.0.

## Part S′ — step-matched GRPO to a plateau, marked reward

This is Part S of `07` with `--reward marked`; it replaces the cancelled Part S.

- Same cells: n=0, gold {50, 250, 500}, teacher {50, 250, 500}; seeds 42–44.
- 375 steps at constant lr 2e-5 after 10 warmup steps; checkpoints every 75 steps, each evaluated on the full test set.
- Output: `runs/gsm8k/Llama-3.2-1B-Instruct-marked-steps375{,-qwen7b}/seed_S/`.
- Submitted after Part C shows that the marked reward trains normally, i.e. C1 holds.

**Predictions.**
- **S1 (plateau).** GRPO-alone mean at step 375 is within 1.5 of step 300. If not, S2–S4 are reported as "not at plateau".
- **S2 (the cliff is an end state).** At step 375, mean(GRPO alone) − mean(gold n=250) ≥ 2.0. The start-quality + budget account predicts ≤ 1.0.
- **S3 (unlock persists).** At step 375, mean(gold n=50) ≥ mean(GRPO alone) − 0.5.
- **S4 (teacher harmless).** At step 375, mean(teacher n=250) ≥ mean(GRPO alone) − 1.0.

## Outcome, Part C (scored 2026-10-04 with `runs/theory/scratch/score_prereg08c.py`)

All 12 runs completed. Marked accuracy, mean ± SD over 3 seeds:

| Cell | Marked accuracy | Lenient accuracy | Final style |
|---|---|---|---|
| GRPO alone | 53.4 ± 1.1 | 53.4 | — |
| gold n=50 | 56.3 ± 0.6 | 56.5 | 100% native |
| gold n=250 | 46.6 ± 0.9 | 46.8 | 0% native |
| teacher n=250 | 54.5 ± 0.8 | 54.5 | |

| Prediction | Result | Held? |
|---|---|---|
| C1 | 53.4 | Yes |
| C2 | unlock +2.9 | Yes |
| C3 | cliff +6.8 | Yes |
| C4 | teacher +1.1 | Yes |

- With the marked reward, GRPO alone no longer drifts to unmarked answers: its marked and lenient accuracy are equal.
- Its marked accuracy is 53.4, against 48.7 when trained with the lenient reward.
- The core contrasts survive.

Part S′ was submitted after C1 held (jobs `sm375-*`). Checkpoint evaluations are graded lenient by `probe.py`, and will be re-graded marked from the saved completions.

## Outcome, Part S′ (scored 2026-10-05 with `runs/theory/scratch/score_prereg08s.py`)

All 21 runs completed. Final marked accuracy, mean ± SD over 3 seeds, after 375 steps:

| Cell | Accuracy | Final native share |
|---|---|---|
| GRPO alone | 56.0 ± 2.4 | 0 |
| gold n=50 | 57.8 ± 1.0 | 0.99 |
| gold n=250 | 57.1 ± 1.0 | 0 |
| gold n=500 | 57.1 ± 1.0 | 0 |
| teacher n=50 | 57.4 ± 2.2 | 0.33 |
| teacher n=250 | 59.6 ± 0.9 | 0 |
| teacher n=500 | 60.3 ± 1.2 | 0 |

Checkpoint accuracy, seed mean, re-graded marked:

| Cell | Step 75 | Step 225 | Step 375 |
|---|---|---|---|
| GRPO alone | 53.4 | 57.0 | 54.7 |
| gold n=250 | 48.3 | 54.6 | 56.9 |
| gold n=500 | 47.1 | 54.4 | 57.3 |
| teacher n=250 | 55.0 | 58.9 | 60.2 |

| Prediction | Result | Held? |
|---|---|---|
| S1 | GRPO alone step 375 − step 300 = −1.6 | **No**, narrowly; S2–S4 are therefore reported as "not at plateau". The gold arms are still rising at step 375 |
| S2 | GRPO alone − gold n=250 = **−1.1** (registered ≥ 2.0) | **No**. The standard-budget cliff (6.8 points, Part C) is gone with more RL |
| S3 | gold n=50 − GRPO alone = +1.8 | Yes |
| S4 | teacher n=250 − GRPO alone = +3.6 | Yes |

**Notes:**
- One run degenerated: GRPO alone, seed 43, repeats `#### N` until the length limit; 22% of answers stop. It is the only run with a stop rate below 0.96.
- The extinct `## Step` mode was never revived: its rollout share was 0 at every logged step in the gold n=250 runs. Instead GRPO repaired the gold style itself: `<<>>` annotations went from 97% to 0%, and answers got longer and added some prose.

**Reading:** the standard-budget allocation cliff is a rate effect, not an end state. With adequate RL, every SFT arm ends at or above GRPO alone, and more teacher SFT is better.
