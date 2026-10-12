# 13 — Measurement hygiene and the censored-transport gate (registered 2026-10-12, before these runs)

## Context
**The round-3 review** (`runs/reviews/round3/track2_icml_assessment.md`) found five problems:
- the 375-step runs never reached a plateau;
- greedy evaluation of the same weights moves by up to 2.6 points across pipelines;
- `modes.classify` merges `## Step` with `\boxed`;
- the marked reward credits a `#### N` repeated until truncation;
- the held-out-NLL result is the dose curve.

**The candidate direction** is censored off-sample transport. On-policy GRPO changes the log-probability of a behaviour it never samples, and the starting policy sets the sign of that change. Its prediction: the change is upward from the untrained instruct model and downward after SFT on a competing style.

**This registration has three parts:**
- **H:** fixes the measurement problems;
- **E1:** measures that change on existing checkpoints;
- **E2:** runs a pilot that decides whether we pursue the direction.

## Part H — measurement changes (no predictions)
Changes to `route_sft_grpo.py`, `modes.py` and `probe.py` (commit `b476c0e`). Two smoke tests (4 and 6 GRPO steps, seed 42, 16 test questions, `dr_grpo`) checked the code; their outputs are not used.

**Disclosed before registration.** In the 6-step smoke test (untrained start, censoring `## Step`, no `## Step` rollouts sampled), the 16-prompt panel mean for `## Step` moved from −11.04 to −10.66. One run of 6 steps is not evidence, and it is recorded here because it was seen before the predictions below were fixed.
- `modes.classify(text, split_native=True)` splits `native` into `step` (`## Step`) and `boxed`.
- `--reward_needs_stop`: an answer earns reward only if the completion ends (EOS) before the length limit.
- `--censor REGEX`: matching GRPO rollouts get reward `None`. TRL 1.14 excludes them from the group baseline (nan-mean and nan-std) and sets their advantage to 0. The matched behaviour is therefore never reinforced or penalised, whatever its sampled rate.
- `--log_rollouts`: every rollout is written to `<method>/grpo/rollouts.jsonl` (step, example id, split mode, reward, correct, stopped, censored, length, text).
- `--panel_prefix P`: the panel. It measures ℓ_P(x), the teacher-forced log p(completion starts with P | x) at temperature 1, on the first 256 test prompts. It is recorded at GRPO step 0 and every 10 steps in `<method>/grpo/panel.jsonl`.
- `--eval_temperature T --eval_k k`: test accuracy as pass@1, the mean over k samples per question. New runs report pass@1 at T = 0.6 with k = 4 as their accuracy.
- **Classifier audit (CPU).** We draw 300 final test completions (`random.Random(13)`) from the `08` Part S′ runs and hand-label them `step` / `boxed` / `gold` / `prompt` / `other`. We report the confusion matrix against `classify(split_native=True)`.

## Part E1 — never-sampled drift in existing checkpoints (inference only, about 1 GPU-hour)
**Adapters.** All come from `08` Part S′, at 375 steps with the marked reward, uncensored, seeds 42–44.
- GRPO alone, gold n=250, gold n=500 and teacher n=250.
- Checkpoints at steps 75, 150, 225, 300 and 375.
- Step 0 is the untrained model (GRPO alone) or the run's SFT adapter (`runs/gsm8k/Llama-3.2-1B-Instruct[-qwen7b]/seed_S/random_sft_N/sft`).

**Measure.** `probe.py --prefix "## Step"` at temperature 1 on all 1319 test prompts. Primary: m(t) = mean ℓ over the first 256 prompts (the E2 panel). Secondary: all 1319 prompts.

**Prediction.** The gold n=250 and n=500 runs never sample `## Step` (`08` S′: at most 1 native rollout in 24,000 per run).
- **E1-a:** pooling these 6 runs, the mean of m(375) − m(0) is negative and at least 2 SE from 0, where SE is the SD of the 6 per-run drifts / √6.
- **Reported, not predicted:** GRPO alone, whose drift is confounded by sampling; and teacher n=250.

E1 does not decide anything. It shows whether a never-sampled drift exists in data we already have.

## Part E2 — censored pilot: the gate (about 12 GPU-hours)
**Runs.** `unsloth/Llama-3.2-1B-Instruct`, LoRA as in all earlier runs. 2 starts × 2 loss types × 3 seeds (42, 43, 44) = 12 runs.

| Start | Method | Starting adapter | GRPO prompts |
|---|---|---|---|
| untrained | `full_grpo` | — | 1000 |
| gold n=250 | `random_sft_250` | reused, `runs/gsm8k/Llama-3.2-1B-Instruct/seed_S/random_sft_250/sft` | the other 750 |

- **Loss types:** `--grpo_loss_type grpo` and `dr_grpo`.
- **GRPO settings:**
  - 150 steps; 8 prompts × 8 samples per step;
  - lr 2e-5, with 10 warmup steps then linear decay to 0;
  - T 0.8, top-p 0.95, completions up to 512 tokens;
  - `--reward marked --reward_needs_stop`;
  - `--censor "## Step"`;
  - `--log_rollouts`;
  - `--panel_prefix "## Step" "## Solution" "Solution:"`.
  - The last two panel prefixes are placebo openings that we report only.
- Checkpoints every 25 steps; a preempted run resumes from its last checkpoint. Measurements repeated after a resume are deduplicated by keeping the last row per step.
- Output: `runs/gsm8k/Llama-3.2-1B-Instruct-censored-{grpo,dr_grpo}/seed_S/`.

**Measures.**
- For run r, the drift is D_r = m_r(150) − m_r(0), where m is the mean panel ℓ for `## Step` over the 256 prompts.
- For each start: D̄ is the mean over its 6 runs (both loss types), with SE = SD/√6.
- Per start and loss type: D̄ and SE over 3 runs.

**Decision rules.** These are the round-3 kill criteria K1 and K2, fixed here.
- **K1 (no start-dependent off-sample drift):** fires if |D̄| < 2 SE for both starts, **or** if D̄(untrained) and D̄(gold-250) have the same sign.
- **K2 (length-normalisation artefact):** fires if, for every start with |D̄| ≥ 2 SE, the drift under `grpo` is ≥ 2 SE (3-run SE) while under `dr_grpo` it is < 2 SE and less than half the `grpo` drift in absolute value.
- **Consequence.**
  - **If K1 or K2 fires:** we drop the ICML pivot. Next we write the COLM/TMLR paper (`track1_venues_narrative.md`) with the Part H measurement changes.
  - **If neither fires:** we register the explanatory and predictive tiers (round-3 E3) as `14` before running them.

**Reported, not predicted:**
- the sign predicted by the transport hypothesis (D̄(untrained) > 0 > D̄(gold-250));
- the number of censored rollouts per run (the entry rate);
- the placebo-prefix drifts;
- the stop rate, entropy and clip ratio;
- final test pass@1 at T = 0.6 with k = 4.

## Scoring
`runs/theory/scratch/score_prereg13.py`. Runs that diverge (stop rate below 0.5 over the last 20 steps) are reported and left out of the gate, as long as at least 2 runs per start and loss type remain. Otherwise the gate is rerun with new seeds.

## Outcome, Part E1 (scored 2026-10-12 with `runs/theory/scratch/score_prereg13.py`)

All 69 adapters and the untrained model were probed on the 1319 test prompts. Drift is m(375) − m(0) in nats on the 256 panel prompts, per seed (42 / 43 / 44):

| Arm | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| gold n=250 | −9.49 | −8.03 | −4.82 |
| gold n=500 | −4.53 | −5.45 | −3.24 |
| GRPO alone (from untrained) | −5.81 | −5.13 | −6.77 |
| teacher n=250 | −2.69 | −0.28 | −3.60 |

| Prediction | Result | Held? |
|---|---|---|
| E1-a | gold drift −5.93 ± 0.96 (SE, n=6); −5.92 ± 0.95 on all 1319 prompts | Yes |

**Not registered (post hoc).** We ran the same probe for two placebo openings that no run produces: `## Solution` and `Solution:` (`runs/theory/scratch/e1_placebo13.py`). Drift, seed mean:

| Arm | `## Step` | `## Solution` | `Solution:` |
|---|---|---|---|
| gold n=250 | −7.5 | −9.1 | −7.1 |
| gold n=500 | −4.4 | −6.4 | −5.8 |
| GRPO alone | −5.9 | −12.4 | −14.1 |
| teacher n=250 | −2.2 | −5.5 | −1.6 |

**Reading.**
- In these uncensored 375-step runs, every opening the policy does not produce loses probability, in every arm, including GRPO from the untrained model. `## Step` usually loses less than the placebos.
- The registered E1 drift is therefore not specific to the extinguished style. It is consistent with generic sharpening (squeezing) of unsampled openings.
- The transport hypothesis's upward drift from the untrained start does not appear in these runs. E2, which censors the target and also uses `dr_grpo`, remains the registered test.
