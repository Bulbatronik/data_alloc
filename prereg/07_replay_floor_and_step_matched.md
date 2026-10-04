# 07 — Replay-floor gate and step-matched RL control (registered 2026-10-04, before any of these runs)

**Context.** Round-2 reviews are in `runs/reviews/round2/`.
- **Part G** is the cheap gate for building a method. It tests the dynamics that any "keep the own style reachable" method relies on (`C_theory.md`, T3).
- **Part S** is the foundation control the experimental review asked for (`B_experiments.md`, item 1). Is the gold cliff an end state, or a rate effect of short RL?

Common to both:
- Model: Llama-3.2-1B-Instruct; GSM8K pool of 1000; SFT lr 1e-4.
- w0: the exact P(answer starts with `## Step`) at T=0.8 on the adapter's GRPO bucket, measured with `probe.py --prefix`.

## Part G — own-style replay floor

**Model** (C's T3, tabular style logit). Suppose a fraction ρ of the SFT traces are own-style (`## Step`) traces and the rest are gold.
- There is a critical fraction ρ_c = λ₀/(a+λ₀).
- Above it, w0 converges to a plateau w*(ρ) = ρ − (1−ρ)λ₀/a that does not depend on the dose n.
- Below it, w0 keeps decaying with n.
- Hand fit to the existing rescue data (k = 0/2/5/20 at n=250): λ₀ ≈ 0.034, a ≈ 0.47, so ρ_c ≈ 7%.

**Runs.**
- SFT with `--sft_traces mix:k:runs/gsm8k/Llama-3.2-1B-Instruct/traces/native.jsonl`, k = round(ρ·n).
- ρ ∈ {4%, 12%, 16%} and n ∈ {250, 500, 750}, seeds 42 and 43.
- k per cell:

  | ρ | n=250 | n=500 | n=750 |
  |---|---|---|---|
  | 4% | 10 | 20 | 30 |
  | 12% | 30 | 60 | 90 |
  | 16% | 40 | 80 | 120 |

- n=250 and n=750 are SFT only. n=500 is SFT followed by standard GRPO (1 epoch, lr 2e-5, full test set).
- Output: `runs/gsm8k/Llama-3.2-1B-Instruct-replay/rho<ρ>/seed_S/`.

**Predictions.**
- **G1 (plateau at ρ=12%).** In every seed, w0 at n = 250, 500 and 750 lies in [2.5%, 10%], and max/min across the three doses is ≤ 3.
- **G2 (plateau at ρ=16%).** In every seed, w0 lies in [5%, 20%] at all three doses, and max/min ≤ 3.
- **G3 (decay below ρ_c, at ρ=4%).** In both seeds, w0(n=750) < w0(n=250)/5 and w0(n=750) < 0.3%.
- **G4 (recovery).** GRPO from the n=500 adapters:
  - ρ = 12% and 16%: recovery in at least 3 of 4 runs.
  - ρ = 4%: 0 of 2.
  - "Recovered" means more than 50% of final greedy answers are `native`.

**Gate decision.** Build the method (forced own-mode openings) only if G1–G3 hold, or if they miss by at most one cell, and G4 holds. Otherwise report the replay model as refuted and do not build on it.

## Part S — step-matched GRPO, run to a plateau

**Runs.**
- GRPO for a fixed 375 optimizer steps (8 prompts × 8 samples per step), cycling the GRPO bucket.
- Constant lr 2e-5 after 10 warmup steps (`--grpo_steps 375 --grpo_lr_scheduler constant_with_warmup --grpo_warmup_steps 10`).
- Adapters saved every 75 steps; each checkpoint evaluated greedily on the full test set (1319).
- Cells:
  - n = 0 (GRPO alone);
  - gold n ∈ {50, 250, 500};
  - teacher (Qwen2.5-7B-Instruct traces) n ∈ {50, 250, 500}.
- Seeds 42, 43, 44.
- SFT adapters are reused from the existing curves (`--reuse_from`), so the start points are identical.
- Output: `runs/gsm8k/Llama-3.2-1B-Instruct-steps375{,-qwen7b}/seed_S/`.

**Competing predictions.**
- Style account: the cliff is an end state.
- Start-quality + budget account (`E_alternatives.md`): non-recovered runs repair towards GRPO-alone level as RL continues.

**Registered predictions.**
- **S1 (plateau).** For GRPO alone, the seed-mean accuracy at step 375 is within 1.5 points of step 300. If this fails, S2–S4 are reported as not at plateau.
- **S2 (the cliff is an end state).** At step 375, mean(GRPO alone) − mean(gold n=250) ≥ 2.0. The start-quality account predicts ≤ 1.0.
- **S3 (no loss from a small dose).** At step 375, mean(gold n=50) ≥ mean(GRPO alone) − 0.5.
- **S4 (a good teacher stays harmless).** At step 375, mean(teacher n=250) ≥ mean(GRPO alone) − 1.0.

## Outcome
(appended after the runs)
