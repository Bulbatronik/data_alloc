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

## Outcome
(appended after the runs)
