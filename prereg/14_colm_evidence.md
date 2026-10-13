# 14 — The paper's contrasts under Part H measurement, on GSM-Symbolic, and on MATH (registered 2026-10-13, before these runs)

## Context
- **Decision so far.** In `13`, K1 fired, so the censored-transport pivot is dropped. The next step is the COLM/TMLR paper (`runs/reviews/round3/track1_venues_narrative.md`).
- **The problem.** The paper's headline contrasts were all measured with greedy batched evaluation. That evaluation moves the same weights by up to 2.6 points across pipelines. For example, `08` S′ step 375, GRPO alone: 56.0 in the training job versus 54.7 in `probe.py`.
- **Before writing, we re-measure the contrasts four ways:**
  - **A:** with sampled pass@1;
  - **B:** on templated variants released after the model;
  - **C:** whether the budget effect also appears on MATH;
  - **D:** whether the Geometry3K test answers can be recovered without the diagram.

**Measurement.** Everything below uses `probe.py` with:
- T 0.6, top-p 0.95, top-k 0;
- marked grading (`--reward marked`).

Accuracy means pass@1, the mean over k samples per question.
- **Primary metric:** marked pass@1.
- **Secondary metric:** "marked and stopped", where the answer must also finish before the length limit.

**Contrasts.**
- **Definition.** A contrast X − Y is the difference of the two cells' seed means, in points.
- **SE** is √(s_X²/n_X + s_Y²/n_Y), where s is the SD over seeds.
- **"≥ 2 SE" is a descriptive bar, not a significance test.** With 3 seeds, the Welch df is about 2–4, so the bar corresponds to a one-sided p of about 0.06–0.09. The paper will not call such a contrast "significant".
- **Best-vs-best** uses, for each cell, the highest seed-mean pass@1 over the evaluated steps. This is an oracle choice of checkpoint on the test set. Its upward bias is largest for the noisiest cell, which is usually GRPO alone. It is a point estimate, with no SE.
- **Non-inferiority bars (A3, B2, C2).** We also report d − 2 SE. The paper says the cliff "closes" only if d − 2 SE ≥ −2.0. Otherwise it says "no cliff detected, but one of more than 2 points is not excluded".
- **Completeness.** A cell is scored only if it has all its registered seeds and every evaluation has its full row count (A: 1319 × 4; B: 5000; C: 1697 × 4). Each (cell, seed, step) appears once. Otherwise the prediction is reported as "not scored".

**Not run.** X4 (the grader test with a standard prompt) is dropped. The round-4 literature audit (`runs/reviews/round4/x0_budget_audit.md`) found no paper among 20 that states it uses a lenient last-number grader. The grader result therefore stays a caveat about our own setup.

## Part A — re-scoring on GSM8K test (inference only, ≈ 21 GPU-h)

**Adapters** (99, plus the untrained model):

| Set | Cells | Seeds | Steps |
|---|---|---|---|
| `08` C, 1 GRPO epoch (`-marked{,-qwen7b}`) | GRPO alone; gold n=50, n=250; teacher n=250 | 42–44 | final |
| `08` S′, 375 steps (`-marked-steps375{,-qwen7b}`) | GRPO alone; gold n=50, n=250, n=500; teacher n=50, n=250, n=500 | 42–44 | 75, 225, 375 |
| SFT starts of `08` S′ (`Llama-3.2-1B-Instruct{,-qwen7b}/seed_S/random_sft_N/sft`) | gold and teacher n=50, n=250, n=500 | 42–44 | 0 |
| `12`, R1-distill, 375 steps (`-r1-marked-steps375`) | GRPO alone; R1 n=250, n=500 | 42, 43 | 375 |

**Measure.**
- All 1319 test questions, k = 4.
- Up to 512 new tokens; 2048 for `12`, as in its training.
- Output: `runs/probes/prereg14_a/`.

**Predictions** (greedy reference values from `08` in brackets; "job" means the training job's own evaluation):
- **A1 (cliff at the standard budget, `08` C).** GRPO alone − gold n=250 ≥ 2.0 and ≥ 2 SE [+6.8].
- **A2 (step-matched cliff inside `08` S′, step 75).** GRPO alone − gold n=250 ≥ 2.0 and ≥ 2 SE [+5.1, `probe.py`].
- **A3 (the cliff closes, `08` S′).** Gold − GRPO alone ≥ −1.0, for n=250 and for n=500, both:
  - at step 375 [+2.2 / +2.6 in `probe.py`; +1.1 / +1.1 in the job];
  - best-vs-best [−0.1 / +0.3, `probe.py`].
- **A4 (teacher benefit persists, `08` S′).** Teacher − GRPO alone, for n=250 and for n=500:
  - at step 375: ≥ 2.0 and ≥ 2 SE [+3.6 / +4.3 in the job; +5.5 for n=250 in `probe.py`];
  - best-vs-best: ≥ 2.0 [+3.2 for n=250, `probe.py`; `08` does not tabulate n=500 by step].

**Reported, not predicted:**
- every cell at every evaluated step (paper figure F1);
- `08` C gold n=50 vs GRPO alone;
- the `12` R1 cells;
- the secondary metric;
- the per-seed spread of each cell.
- **Sensitivity.** The contrasts recomputed without adapters whose evaluation stop rate is below 0.5. Two such runs are known: `08` S′ GRPO alone seed 43 (22% of answers stop) and `12` GRPO alone seed 42 (41%).

## Part B — GSM-Symbolic (inference only, ≈ 7 GPU-h)

**Data.** `apple/GSM-Symbolic`, `main` configuration (`--dataset gsm_symbolic`), all 5000 questions, k = 1.
- 100 GSM8K-test templates × 50 instances with new names and numbers.
- **Release:** October 2024, after Llama-3.2's data cutoff (December 2023) and release (September 2024).
- **Not a clean post-cutoff set.** The templates are GSM8K test questions from 2021. What is new is the names and numbers.
- **Not our training pool.** The templates come from GSM8K test, so none is one of our training problems.

**Adapters** (36, plus the untrained model):
- `08` C, all 4 cells × 3 seeds;
- `08` S′ GRPO alone, gold n=250, teacher n=250 and teacher n=500, at steps 225 and 375, × 3 seeds.

Output: `runs/probes/prereg14_b/`.

**Predictions:**
- **B1.** The A1 contrast on GSM-Symbolic is ≥ 2.0 and ≥ 2 SE.
- **B2.** The A3 contrast for n=250 on GSM-Symbolic is ≥ −1.0, both at step 375 and best-vs-best over steps 225 and 375.
- **B3.** The A4 contrasts (n=250 and n=500) at step 375 on GSM-Symbolic are each ≥ 2.0 and ≥ 2 SE.

**Reported, not predicted:**
- the accuracy drop from GSM8K to GSM-Symbolic for every cell;
- the same drop restricted to the 100 GSM8K test questions the templates come from;
- the sensitivity check of Part A.

## Part C — the budget effect on MATH levels 1–3 (≈ 25 GPU-h)

**Runs.** `unsloth/Llama-3.2-1B-Instruct`, `--dataset math`: the same pool of 1000 numeric-answer problems as the earlier MATH runs (`--data_seed 1234`).
- **Cells:** GRPO alone (1000 prompts) and gold n=250 (750 prompts).
- **SFT adapter** reused from `runs/math/Llama-3.2-1B-Instruct/seed_S/random_sft_250/sft`.
- **Seeds:** 42, 43 and 44.

**Earlier MATH runs, for reference** (`runs/math/Llama-3.2-1B-Instruct/seed_*/results.csv`; 1 epoch, lenient reward, greedy):
- GRPO alone: 51.0 (seeds 50.9 / 50.7 / 51.4);
- gold n=250: 43.0 (44.1 / 41.7 / 43.2);
- gold n=250 SFT alone lowers the untrained model from 48.0 to 38.4.

**GRPO settings.** These match `13`'s E2, run to 375 steps.
- 8 prompts × 8 samples per step, as micro-batches of 4 × 16 gradient-accumulation steps, as in the earlier MATH runs;
- lr 2e-5, 10 warmup steps, then linear decay to 0;
- T 0.8, top-p 0.95;
- completions up to 1024 tokens, as in the earlier MATH runs, since 512 truncates MATH solutions;
- `loss_type grpo`;
- `--reward marked --reward_needs_stop`;
- `--log_rollouts`;
- checkpoints every 25 steps; preempted runs resume.

Output: `runs/math/Llama-3.2-1B-Instruct-marked-steps375/seed_S/`.

**Evaluation.** All 1697 MATH levels 1–3 numeric test problems, k = 4, up to 1024 new tokens.
- **Scored.** `probe.py` at steps 75, 225 and 375 (`checkpoint-*`), written to `runs/probes/prereg14_c/`. This is a single pipeline for every scored number.
- **Reported only.** The training job's own evaluations: the untrained model, after SFT, and step 375 (`--eval_temperature 0.6 --eval_k 4 --eval_max_new_tokens 1024 --max_eval_samples -1`).

**Exclusion.** A run whose GRPO stop rate over the last 20 steps is below 0.5 is reported and excluded.
- After an exclusion, a cell needs at least 2 runs to be scored.
- If a cell has fewer than 2 runs left, Part C is rerun with new seeds, as in `13`.

**Predictions:**
- **C1 (step-matched cliff, step 75).** GRPO alone − gold n=250 ≥ 2.0 and ≥ 2 SE.
- **C2 (the cliff closes).** Gold n=250 − GRPO alone ≥ −2.0, both at step 375 and best-vs-best over steps 75, 225 and 375.
  - The bar is looser than A3's −1.0 because gold SFT costs more before RL on MATH than on GSM8K:
    - MATH: 9.6 points (48.0 → 38.4, greedy);
    - GSM8K: 1.8 points (36.2 → 34.4, pass@1, `13` E2).

**Reported, not predicted:**
- training reward;
- stop rate;
- rollout styles;
- the secondary metric;
- the training job's evaluations.

## Part D — Geometry3K test without the diagram (inference only, < 1 GPU-h)

**Question.** Can the test answers be recovered without the diagram? This tests for memorisation by the VLM.
- `10` M2 answered it on *training* items: 7.5% text-only.
- The same model, with the image, scored 22.8% on train items and 0.25 on test items (`10`; `runs/vlm/geo3k/m0/student_test.*`).

**Measure.** Untrained `Qwen/Qwen2.5-VL-3B-Instruct` on the 601 test items, with `--input text --prompt_style boxed_user`.
- T 0.8, top-p 0.95, 8 samples per item, up to 1024 tokens. These are the `10` sampling settings, not the Part A ones, so the results compare with `student_test`.
- Output: `runs/vlm/geo3k/m0/student_test_text.s0.jsonl`.
- Both the text-only and the image samples are re-graded with the current marked Geometry3K grader.

**Prediction:**
- **D1.** Mean text-only accuracy on test is ≤ 0.10, and ≤ half the image accuracy on the same items. At the known image accuracy of about 0.25, the 0.10 bar is the binding one.

## Scoring
`runs/theory/scratch/score_prereg14.py`, written before any Part A–D output was read.
- It was dry-run on synthetic outputs of the registered size.
- A prediction is "HELD" only if every one of its sub-conditions is scorable and holds.
- If a part's runs fail for technical reasons, that part is rerun.
- The registration was checked by a skeptical reviewer subagent before it was pushed. Its corrections are included above:
  - MATH token length;
  - completeness checks;
  - a single pipeline for Part C;
  - the wording rule for non-inferiority;
  - the sensitivity check.
