# 09 — Method gate for forced own-mode openings, and a second latent mode (registered 2026-10-04, before any of these runs)

**Status.** `07` Part G refuted its replay model, so under its registered rule nothing is built on it. This file replaces 07's decision rule for the method with a gate on what forced own-mode openings in GRPO actually need:
- the forced own mode is competent;
- forced answers do not slide back into the resident style;
- the mode is not already reachable by REFT-style top-20 first-token forcing.

All accuracies use the marked grading of `08`.

## Part A — Llama-3.2-1B-Instruct, own mode `## Step`

**Cells.**
- Gold SFT, n=250, seeds 42–44: `runs/gsm8k/Llama-3.2-1B-Instruct/seed_S/random_sft_250/sft`.
- R1-distill SFT, n=500, seeds 42–44: `runs/gsm8k/Llama-3.2-1B-Instruct-r1/seed_S/random_sft_500/sft`.
- Untrained model: reference only.

**Measurements** (`probe.py`):
- 256 GRPO-bucket prompts × 8 samples, T=0.8, top-p 0.95. For the untrained model, 256 pool prompts.
- Two sampling conditions:
  - natural sampling → p_res (overall accuracy; almost all answers are in the resident style at these doses);
  - sampling forced to open with `## Step` (`--force_prefix`) → p_own.
- Hybrid rate: the share of forced answers that contain the resident style's marker. For gold that is a `<<…=…>>` annotation; for R1 it is `</think>`.
- REFT reach: the share of prompts on which the first token of `## Step` is among the model's top-20 first tokens (`runs/theory/scratch/g09_first_token_rank.py`).
- Generation limit: 512 tokens for gold cells, 2048 for R1 cells.

**Gate rule.** Build forced-own-opening GRPO only if all three hold on BOTH damaged cells (seed means):
- **A1.** p_own − p_res ≥ 0.05.
- **A2.** Hybrid rate ≤ 50%.
- **A3.** `##` is in the top-20 on at most 50% of prompts. Otherwise REFT already reaches the mode, and the method is not novel against it.

**Registered expectation (not part of the gate).** Earlier lenient data on other gold adapters had Δp ≈ +0.2 but hybrid rates of 46–89% (`runs/theory/theory_v2.md` §8). So we expect A1 to hold for gold, and A2 to be at risk for gold.

## Part B — Qwen2.5-Math-1.5B on MATH (levels 1–3, numeric), own mode: code-assisted reasoning

**Setup.**
- Prompt: Qwen-Math's native `boxed` style.
- Pool: 1000 MATH training problems; test: MATH levels 1–3 test set.
- SFT traces: MATH reference solutions, which contain no code.

**Measurements.**
- **B-untrained.** 400 test prompts × 4 samples, T=0.8, 1024 tokens:
  - code share: completions containing a fenced code block;
  - p_code: accuracy of code answers;
  - p_NL: accuracy of the other answers;
  - accuracy when forced to open with a ```` ```python ```` block.
- **B-scan.** SFT only on reference solutions, n ∈ {50, 100, 250, 500, 1000}, seed 42, lr 1e-4. Code share and accuracy on 256 test prompts × 4 samples per adapter.

**Predictions.**
- **B1 (a better latent mode exists).** Untrained code share ≥ 10%, and p_code − p_NL ≥ 0.05.
- **B2 (SFT suppresses it).** Code share after SFT is below 1% at n=1000 and decreases from n=250 to n=1000.

**Use.** Mode B becomes the method's second test mode only if B1 holds. If B1 fails, it is dropped and reported.

## Outcome (scored 2026-10-05 with `runs/theory/scratch/score_prereg09.py`)

**Part A.** Seed means, marked grading:

| Cell | p_res | p_own | A1 (gap) | A2 (hybrid rate) | A3 (top-20 share) |
|---|---|---|---|---|---|
| gold n=250 | 0.39 | 0.56 | +0.17 ✓ | 0.62 **✗** | 0.00 ✓ |
| R1 n=500 | 0.50 | 0.65 | +0.15 ✓ | 0.04 ✓ | 0.00 ✓ |
| untrained | 0.46 | 0.69 | | | 0.82 |

- **Gate rule (all three on both cells): failed**, on gold A2. The method is not built under this registration.
- Not registered: the realistic R1 cell passes all three checks. After SFT, the own mode is out of REFT's top-20 reach on every prompt, against 82% for the untrained model.

**Part B (Qwen2.5-Math-1.5B, MATH levels 1–3).**
- **B1: yes.** The untrained code share is 0.39. Code answers are far more accurate than natural-language ones (0.56 vs 0.13). Forced ```` ```python ```` openings score 0.52.
- **B2: yes.** NL-only SFT drives the code share from 0.42 (n=50) to 0.18 (n=100), 0.017 (n=250), 0.002 (n=500) and 0 (n=1000).
- Not registered: accuracy after SFT *rises* even as code dies (0.30 untrained → 0.57 at full SFT), because the SFT'd natural-language style becomes competent.
