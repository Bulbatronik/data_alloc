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
