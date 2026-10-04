# 03 — Long-CoT teacher arm through GRPO (registered 2026-10-02, before any of these runs)

## Setup
- SFT traces: DeepSeek-R1-Distill-Qwen-1.5B traces on the 1000-problem pool (`runs/gsm8k/teachers/r1_1.5b.jsonl`), using the first correct trace per problem. 931/1000 problems are covered.
- Pipeline: SFT (lr 1e-4) on n problems, then 1-epoch GRPO on the other 1000 − n.
- Lengths: SFT length 2304, GRPO completions 2048 tokens, evaluation 2048 tokens.
- Evaluation: full test set (1319).
- Doses: n = 0 (GRPO alone, same long settings), 100, 250, 500.
- Seeds:
  - Llama-3.2-1B-Instruct: 42, 43, 44
  - Llama-3.2-3B-Instruct: 42, 43
  - SmolLM2-1.7B-Instruct: 42, 43
- Output: `runs/gsm8k/<student>-r1/seed_<s>/`.

**Own-style share ŵ** (`probe.py --own_style`).
- Library built from each untrained student against its R1 full-SFT adapter (seed-42 scan). Λ = ln 10⁴, 64-token openings.
- Measured after training, on each run's own SFT adapter and on its GRPO prompts.

**Final style** (pre-specified surface rule, checked on the SFT-only scans).
- An answer is R1-style if it contains `</think>`, or contains `\boxed{` without `## Step`; otherwise it is own-style.
- On untrained models the rule gives 0% R1 for both Llamas and 10.6% for SmolLM2, whose native style sometimes uses `\boxed`.
- "Recovered" = more than 50% of final greedy answers are own-style.

## Known before registering (seed-42 SFT-only scans, 300-item evaluation)

ŵ at n = 100 / 250 / 500:

| Student | ŵ | Library mass |
|---|---|---|
| Llama-1B | 27% / 0.88% / 0.023% | 0.76 |
| Llama-3B | 25% / 0.59% / 0.020% | 0.70 |
| SmolLM2 | 65% / 9.0% / 0.53% | 0.16 (weak) |

Fitness-gap proxy Δp = untrained accuracy − R1 full-SFT accuracy:

| Student | Untrained | R1 full SFT | Δp |
|---|---|---|---|
| Llama-1B | 45.0 | 44.0 | +1.0 |
| Llama-3B | 80.7 | 76.3 | +4.4 |
| SmolLM2 | 47.0 | 42.0 | +5.0 |

After-SFT accuracy at n ≥ 100:
- Llama-1B: 43–51 (neutral)
- Llama-3B: 72–77
- SmolLM2: 39–44

## Predictions

**R1 — survival band (per run, using ŵ measured on that run's SFT adapter).**
- ŵ < 0.3% ⇒ not recovered.
- ŵ ≥ 10% ⇒ recovered.
- In between: no prediction.
- Registered: at least 90% of runs in the two decisive zones go as called.
- Expected from the seed-42 values:
  - n=100 recovers for all three students;
  - n=500 does not recover for both Llamas;
  - n=250 is in the band for all three students;
  - SmolLM2 n=500 is in the band.

**R2 — cliff where the surviving style is worse (Llama-3B, Δp = +4.4).**
- Mean final(n=500) ≤ mean final(n=0) − 2.0.
- Mean final(n=100) ≥ mean final(n=0) − 1.5.

**R3 — SmolLM2 (Δp = +5.0).**
- Mean final(n=100) ≥ mean final(n=0) − 1.5.
- No accuracy prediction at n=500: ŵ is in the band, and with gold traces SmolLM2 repaired the foreign style within RL.

**R4 — no cliff without a fitness gap (Llama-1B, Δp ≈ +1).**
- At every dose, mean final ≥ mean final(n=0) − 2.0, whether or not the own style survives.

## Scoring
- R1: per run.
- R2–R4: on seed means.
- Runs that diverge or time out are reported and not scored.

## Outcome (scored 2026-10-04 with `runs/theory/scratch/score_prereg03.py`; lenient grading as registered)

All 20 runs completed. Mean final accuracy by n:

| Student | n=0 | n=100 | n=250 | n=500 |
|---|---|---|---|---|
| Llama-1B (3 seeds) | 53.2 | 57.4 | 51.0 | 51.0 |
| Llama-3B (2 seeds) | 81.1 | 82.8 | 79.6 | 79.0 |
| SmolLM2 (2 seeds) | 50.1 | 51.1 | 49.0 | 44.2 |

| Prediction | Result | Held? |
|---|---|---|
| R1 | 12/12 decisive runs as called. n=100 recovers (ŵ 20–65%); n=500 does not (ŵ ≤ 0.5%). Every n=250 run (in the band) ended non-recovered | Yes |
| R2 | Llama-3B final(500) − final(0) = −2.1 (≤ −2.0); final(100) − final(0) = +1.7 | Yes, by 0.1 point |
| R3 | SmolLM2 final(100) − final(0) = +1.1 | Yes |
| R4 | Llama-1B −2.2 at n=250 and n=500 (registered: no worse than −2.0) | **No** |

**Reading:**
- The Δp proxy (untrained minus full-SFT accuracy) did not separate the students. It predicted a cliff for Llama-3B (+4.4) and none for Llama-1B (+1.0), but both show about −2.
- Under marked grading R2 narrowly fails (−1.9 vs −2.0); see `07`/`08` for the grading audit.
- The survival rule works. The harm-size proxy does not.
