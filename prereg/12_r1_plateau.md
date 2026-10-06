# 12 — Does the realistic (R1-distill) cliff also vanish with more RL? (registered 2026-10-05, before these runs)

**Context.**
- In `08` Part S′, the gold-trace cliff disappeared at 375 GRPO steps with the marked reward. Gold n=250 ended 1.1 points *above* GRPO alone, while the extinct `## Step` mode was never revived.
- In `03`, the R1-distill arm had a smaller cliff at the standard budget: Llama-1B −2.2 at n=250 and n=500, lenient grading.
- Question: is the realistic case also a rate effect?

**Runs.**
- Llama-3.2-1B-Instruct, reusing the R1-distill SFT adapters of `03` (`--reuse_from runs/gsm8k/Llama-3.2-1B-Instruct-r1/seed_S`).
- Cells: n = 0 (GRPO alone), 250, 500; seeds 42, 43.
- GRPO: 375 steps, constant lr 2e-5 after 10 warmup steps, marked reward, completions and evaluation up to 2048 tokens.
- Checkpoints at steps 225 and 375 evaluated on the full test set.
- Output: `runs/gsm8k/Llama-3.2-1B-Instruct-r1-marked-steps375/seed_S/`.

**Predictions** (rate account, from `08` S′):
- **Q1.** mean final(n=250) ≥ mean final(0) − 1.0, and mean final(n=500) ≥ mean final(0) − 1.0.
- **Q2.** The extinct own mode is not revived: final greedy `## Step` share < 5% at n = 250 and n = 500 in every run.

## Outcome
(appended after the runs)

## Amendment (2026-10-06, implementation only; no Part-12 outcome seen except seed-43 GRPO alone)

Both jobs were preempted several times. Seed 42 lost GRPO-alone progress three times, most recently at step 359 of 375.

`route_sft_grpo.py` now resumes GRPO from the last checkpoint in a method's directory (`get_last_checkpoint`). Checkpoints now hold the full trainer state, and the resubmitted jobs save every 25 steps.

**Deviation.** The checkpoints that already existed (step 225: seed 42 GRPO alone, seed 43 n=250) were saved model-only. Those two runs therefore resume at step 225 with the Adam moments reset, and the LR schedule's 10 warmup steps repeat. All other runs are unaffected.
