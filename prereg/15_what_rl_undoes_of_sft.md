# 15 — What RL undoes of SFT, item by item: help vs damage, a MATH check, a protocol control and a first-gradient gate (registered 2026-10-21, before any Part X / Part 1 run and before any per-item analysis of Parts 0, A, B)

**Status.** Registered by commit + push to `prereg/`.
- The draft text below is the registration. "Draft" and "lead: confirm" wording in it is historical.
- The user's autonomy grant of 2026-10-19 covers registration.

**Scorer and submission script, frozen at registration** in `prereg/scripts/p15/` (SHA-256):
- `p15_mock.py` 5fbdbe5f22b425ddafbe6748aa4de6925e1fe2f1972d0bc417be7d56166bca87
- `p15lib.py` 9c15684cbdd0ec13e00f6cf36034b475ad51fb22a2b16dae97413735606edf0d
- `run_p15_dry.sh` ca4d7ae2f10fe5be3b832e3aadbfd68bc18faba3b72851308df3f3e36234469f
- `score_prereg15.py` 9def2afa975ecfca4ad8769f23fdc6cf7c82ca275da4da5ab21be6b48ee32e31
- `submit_prereg15x.sh` ee57a1d46d6e611ce262fc569fbb8b53d8b27deae0f158ded0f3455f6ce0ff0a

`prereg/16` is scored with its own frozen copy (`prereg/scripts/p16/`, the scorer as of the earlier registration of 16). Part X runs on `mit_preemptable` (requeue + resume) because `mit_normal_gpu` is occupied by prereg 16; everything else in the run commands is unchanged.

**Known power limitation, registered here.** The location check guards against a logit-symmetric world (job 26491169, `runs/theory/scratch/score_prereg15_dry_power_check_out.txt`). It costs power:
- false H-repair under logit symmetry falls from 0.13 to 0.02;
- P(H-repair) for a true repair world with r₀ = 0.6 falls from 0.82 to 0.50;
- a partial world (helped 0.8, hurt 0.5) reads H-repair only 0.37 of the time.

An "intermediate (location-explained)" verdict is therefore not evidence against weak repair.

## Context
- **Contraction on GSM8K.** In `13` E2 (Llama-3.2-1B-Instruct, GSM8K, gold n=250 vs the untrained start, 150 GRPO steps, sampled k=4), RL removes about half of the per-item SFT effect. The errors-in-variables (EIV) retention slope is 0.457 under marked grading and 0.562 format-insensitive (`G1_convergence_e2.py` `rho_item`; reproduced exactly by this scorer).
- **Two open questions:**
  - Q-A: does RL undo SFT's *helpful* per-item effects, or only repair damage?
  - Q-1: is the reversion visible in the first RL gradient?
- **Three problems from the theory section** (`paper/reversion/main.tex`) that the earlier draft did not handle:
  - (i) The one-index null is 1 only for a logit shift, and only in logit units. In probability units it must be simulated, and a tilt alone makes probability-scale retention *fall* with base success (Prop. 1d). So "retention falls with base success" passes under the null.
  - (ii) Splitting items by sign on a noisy half mixes the classes. This pulls b₊ and b₋ toward each other, i.e. toward symmetric reversion.
  - (iii) In E2 and `08` S′, the untrained start trains a *fresh* LoRA (B = 0) while SFT starts *continue* their adapter. The per-item dynamics and the first-step geometry therefore differ by protocol (Remark "asymmetric geometry"; the kernel account needs a shared parameterisation).

## Common conventions (all parts)
**Grading.**
- Primary: format-insensitive, `extract_final_answer(..., marked=False)` + `normalize_number`. This is the pipeline's lenient last-number grader.
- Secondary, reported throughout: marked (`marked=True`).
- The scorer's grader is a port of `route_sft_grpo.answer_is_correct`, and a regression test checks it:
  - identical to the live pipeline function on 63,312 / 63,312 (completion, grader) pairs;
  - the marked regrade equals the stored `correct` field on 94,968 / 94,968 E2 samples.

**Scales.**
- Primary: probability scale.
- Every retention statistic is also reported on the logit scale: logit((c + ½)/(n + 1)) per half and in full, with n the cell's nominal sample count.
- Logit-scale statistics are judged only against their own simulated null, never against 1, because the smoothing biases them (paper App. A).

**Per-item quantities.** For start S, the untrained start U, RL from S (post) and RL alone R:
- D_pre = p_S − p_U and D_post = q_S − q_R.
- Halves H1 and H2 are samples {< k/2} and {≥ k/2} of every evaluation, so both halves contain every adapter. Post-RL samples are a third, disjoint set.

**Estimators.** Every pooled estimate is a ratio of sums of item-weighted moments over arms (and sets), i.e. a stacked, arm-centred estimator.
- **EIV retention b** = Cov(D̂_post, D̄_pre) / Cov(D̂⁽¹⁾, D̂⁽²⁾). This is G1's `rho_item` and the paper's eq. (bhat).
- **Decomposition** (paper eq. decomp, Prop. 7c), cross-fitted over the two half assignments:
  - β_π = [Cov(q_π − p_π⁽¹⁾, D⁽²⁾) + Cov(q_π − p_π⁽²⁾, D⁽¹⁾)] / (2 Cov(D⁽¹⁾, D⁽²⁾));
  - 1 + β_S − β_U = b exactly on the probability scale;
  - β_S < 0 is *undoing* (the SFT start moves against its own effect); β_U > 0 is *convergence* (RL alone moves toward the SFT start).
- **Sign split** (primary): cross-fitted Wald chord, τ = 1/3.
  - In fold f, items are grouped on the probability-scale half-f difference D⁽ᶠ⁾: helped A₊ is D⁽ᶠ⁾ ≥ τ; hurt A₋ is D⁽ᶠ⁾ ≤ −τ; the reference A₀ is |D⁽ᶠ⁾| < τ.
  - Every quantity is measured on the other half and on the post-RL samples.
  - b± = Σ_f [D̄_post(A±) − D̄_post(A₀)] / Σ_f [D̄_meas(A±) − D̄_meas(A₀)].
  - Selection and measurement use disjoint samples, so there is no regression to the mean (Prop. 7d). The chord from A₀ removes a common post-RL offset between the starts.
  - The through-origin ratio Σ_f D̄_post(A±) / Σ_f D̄_meas(A±) (paper Def. 1) is reported as secondary. In the calibration it turns a ±0.03 common offset into a spurious b₊ − b₋ of −0.22 / +0.16, whereas the chord stays within 0.03.
  - Chord decomposition: 1 − b± = u± + c±, with u = −chord(q_S − p_S^meas)/chord(D_meas) the undoing part and c = chord(q_R − p_U^meas)/chord(D_meas) the convergence part.
- **History asymmetry ("memory")**, probability scale only:
  - λ_S is the coefficient of p_U in the EIV linear projection of q_S on (p_S, p_U). λ_R is the coefficient of p_S in q_R on (p_U, p_S); this is the placebo, since RL alone never saw the SFT model.
  - True-score variances come from split halves; cross-moments come from disjoint samples.
  - M = λ_S − λ_R. M > 0 means RL pulls the SFT start's profile back toward the base profile (undoing); M < 0 means RL alone acquires the SFT profile (convergence).
  - **Why this method.** It is consistent for the linear projection under measurement noise, cheap enough for B = 2000, and transparent.
  - **Its weakness.** Nonlinearity alone can make M ≠ 0. So M is judged only against a simulated *symmetric no-asymmetry* null (below), never against 0.
  - T's NPMLE latent model is fitted as the null generator. Its free λ_S − λ_R is printed as a secondary estimate.
  - The logit-scale version is not used: its split-half moment matrix is near singular (the E2 smoke run gave λ ranges of ±300).
- **Base-success bins** (R1_krl_checks): bins are by the untrained start's success on H1: 0, (0, ⅓], (⅓, ⅔], (⅔, 1), and 1.
  - D_pre = p_S (all samples) − p_U (H2); retention per bin = mean D_post / mean D_pre.
  - **Every base-success claim is an excess over the per-bin simulated one-index null:** [ret(hi) − null(hi)] − [ret(lo) − null(lo)].
  - hi and lo are the highest and lowest bins with n ≥ 15 and |mean D_pre| ≥ 2 points, fixed on the observed data.
  - It is reported on both scales.

**Nulls**, simulated through the identical cells and estimators (paper App. B):
1. A 2-D NPMLE prior of (logit p_U, logit p_S) is fitted on the pre-RL counts (17 × 17 grid on [−6, 6]).
2. True successes are drawn per item from its posterior given its own counts.
3. Binomial samples are drawn at the observed per-unit, per-half sample sizes.
4. The same estimator is applied. 200 replicates.

The three null generators:
- **One-index null (primary): ONE link** fitted on the untrained start's own pair (p_U → q_R) and applied to both starts: q = expit(logit p + s_U). This replaces the two-index null with start-specific shifts used in round 4.
- **Affine sensitivity:** one affine-logit link, fitted the same way.
- **Symmetric memory null:** logit q = γ + β·logit p_own + λ₀·logit p_other + τz.
  - It uses the same λ₀ for both starts: the mean of the two freely fitted λ's. γ, β and τ are refitted given λ₀.
  - Asymmetry is detected only if M − median(null M) has a 95% CI excluding 0 **and** M lies outside the central 95% of the null.
- Pooled nulls sum per-arm simulated moments replicate by replicate.

**Uncertainty.** Hierarchical bootstrap, B = 2000, two-sided 95% percentile intervals, for every statistic.
- Each replicate resamples seeds within lineage, then items.
  - A lineage seed is an SFT adapter together with its RL run, resampled jointly.
  - In Part X, the `cont` and `mf` arms of a source share the SFT adapter and are paired.
  - In prereg 16, the same seed draw applies to every evaluation set.
  - Untrained evaluations are not resampled.
- **Item clusters:** GSM-Plus perturbations together with their GSM8K seed question (prereg 16); GSM-Symbolic instances by template.
- **Coverage check** (synthetic dry run §3: 30 datasets, B = 200, true-score estimands):
  - The hierarchical intervals cover b at 0.97–1.00 and b± at 0.97–1.00; they are conservative.
  - An item-only bootstrap with seeds fixed under-covers b± (0.87–0.93). It was considered because repeating a seed inflates per-item noise and so shifts selection counts. It is not used.
  - Bootstrap intervals of selection *counts* (items per side) are not meaningful and are not interpreted.

**Exclusions.**
- A GRPO run with stop rate < 0.5 over its last 20 steps is reported and excluded with its whole lineage (SFT adapter and RL run). For GRPO alone, it is excluded from the RL cells.
- Stop rates come from `rollouts.jsonl` (E2, Part C, Part X; resume replays removed). For the `08` S′ runs, which predate `--log_rollouts`, they are 1 − mean TRL `completions/clipped_ratio` over the logging entries at steps 360 and 370 (steps 351–370).
- A cell with < 2 seeds left is not scored; it is rerun with new seeds, as in `13`.

**Primary configuration.**
- Lenient grading, probability scale, step 375 (Part A), 150 (Part X) or 375 (Part 0).
- Sources pooled over n ∈ {250, 500} (Part A); n = 50 is reported separately.
- **Untrained start for GSM8K:** probe.py (4 samples) pooled with E2's three training-job evaluations (12 samples). These have the same model, prompt, T 0.6, top-p 0.95, top-k 0, 512 tokens and `--data_seed 1234`; gold alignment by example_id has 0 mismatches.
  - Pooling cuts SD(b₊ − b₋) from ≈ 0.075 to ≈ 0.047 (dry run §power).
  - The probe-only version is reported as a sensitivity check.

## Part 0 — MATH L1–3: does per-item contraction replicate? (0 GPU-h; registered after the data existed)
**Data.** `14` Part C: Llama-3.2-1B-Instruct, 1,697 MATH L1–3 numeric test items, untrained vs gold n=250, seeds 42–44, 375 steps.
- Pre-RL: the training jobs' `base_predictions.jsonl` (U, 12 samples) and `random_sft_250/predictions_after_sft.jsonl` (12 samples).
- Post-RL (primary): `runs/probes/prereg14_c/ckpt.jsonl` at steps 75, 225 and 375 (12 samples per arm).
- Independent replication set: the training jobs' step-375 `predictions.jsonl`.
- T 0.6, not the RL temperature.

**Decision rule P0-1** (lenient, probability scale, step 375, probe checkpoints):
- **Contraction replicates:** b ≤ 0.75 **and** b ≤ median(one-index null) − 0.25.
- **Does not replicate:** b > 0.85 **or** b > median(null) − 0.10. Then the reversion line is not extended beyond this registration (no stage-2 study), whatever Part X and Part 1 show, and the paper falls back to the measurement paper.
- **Intermediate:** otherwise. Report it; no decision.

**P0-2** (restated per the theory; lenient): the base-success excess [ret(hi) − null(hi)] − [ret(lo) − null(lo)] is < 0, with its 95% CI excluding 0. This means reversion is concentrated where the base model already succeeds, *beyond saturation*. The logit-scale version is reported.

**Reported, not predicted:**
- b, β_S, β_U, b±, M and bins at steps 75, 225 and 375, on both scales and both graders;
- the replication set;
- pool equalisation: the gold − untrained training-reward gap in steps [350, 375) over [0, 25), averaged over seed pairs. If > 0.3, the kernel account's equalisation premise fails on MATH (as on GSM8K: −0.090 → −0.100).

## Part A — GSM8K help vs damage, decomposition and history asymmetry (0 GPU-h; decisive)
**Data.** `14` Part A (`runs/probes/prereg14_a/`): GSM8K test (1,319 items), T 0.6, k = 4 per adapter.
- Starts: gold and teacher (Qwen2.5-7B-I) traces, n ∈ {50, 250, 500}, seeds 42–44 (`sft.jsonl`); `08` S′ RL checkpoints at steps 75, 225 and 375 (`s75/s225/s375.jsonl`).
- RL alone: `full_grpo` at the same steps. Untrained start: as in the conventions.
- Primary: step 375, gold n ∈ {250, 500} pooled vs teacher n ∈ {250, 500} pooled.

**A1 — contraction beyond the null** (registered, per source; lenient):
- b < median(one-index null b) − 0.10, with the 95% CI of b entirely below that value.
- Reported: logit-scale b vs its null; marked; per arm; steps 75 and 225.

**A2 — help vs damage (the decisive question).** Notation: b₊(T) is the pooled teacher helped-item retention; b₋(G) is the pooled gold hurt-item retention; d = b₊(T) − b₋(G). Lenient, probability scale, Wald chord, τ = 1/3, step 375.

| Verdict | Rule (point estimates) |
|---|---|
| **H-sym** (symmetric reversion) | \|d\| < 0.10 **and** b₊(T) ≤ 0.70 |
| **H-repair** (RL reverts damage more than help) | d ≥ 0.10 **and** b₊(T) ≥ 0.50; sub-label "pure repair" if also b₊(T) ≥ 0.75 |
| **Intermediate** | anything else (including d ≤ −0.10: help undone more than damage, and \|d\| < 0.10 with b₊(T) > 0.70: no contraction of help) |
| **Untestable** | a side has < 40 items per arm, or its D_pre chord is < 3 points |

- **Why these thresholds** (dry run `--dry_run power`, jobs 26375733 and 26376985 (identical seeds and results), 200 datasets per world at Part-A sizes, E2-sized effects, Var_true(D_pre) ≈ 0.04):
  - **Symmetric worlds** (b₀ = 0.4, 0.5 or 0.6, with or without a ±0.03 common offset) give b̂₊ ≈ b̂₋ ≈ b₀ (within 0.045; at b₀ = 0.6, b̂₊ reads 0.557).
  - **Repair-only worlds** (helped effects fully retained; hurt retained r₀ = 0.3, 0.45 or 0.6) give b̂₊(T) ≈ 0.76–0.79 (0.81–0.83 with half-size effects) and b̂₋(G) ≈ r₀. Full retention therefore *reads* about 0.8, not 1, because the noisy split mixes the classes. The draft's H-repair bar b₊ ≥ 0.85 would have misclassified most true repair worlds.
  - With the rule above, every symmetric and every repair-only world is classified correctly with probability ≥ 0.87 (registered-rule table in the power output: symmetric 0.87–0.97, repair-only 0.89–1.00). The binding worlds are b₀ = 0.6 and r₀ = 0.6. τ = 1/3 beats 0.2 and 0.5. A "partial" world (helped 0.8, hurt 0.5) reads H-repair (0.85), usually without the "pure" sub-label.
  - SD(d) = 0.045–0.056, so the **MDE for b₊ − b₋ is 0.13–0.16** (80% power, two-sided 5%).
  - **With half-size SFT effects** (Var_true ≈ 0.02), the worst-case correct classification falls to 0.77 (symmetric b₀ = 0.6) and the MDE is 0.18–0.25. **With the probe-only untrained start** (4 samples) it falls to 0.78, or 0.57 with half-size effects; this is why the untrained start is pooled.
- **Location check** (registered; it can only *downgrade* a verdict to "intermediate"; `split_location_check`).
  - **Why.** Teacher-helped and gold-hurt items sit at different base-success locations, so ceiling and floor effects alone can make d ≠ 0 under symmetric reversion. In the mock end-to-end test, a world symmetric on the logit scale (each start keeps half of its logit effect) gave d = +0.15 on the probability scale and +0.16 on the smoothed-logit scale. The fixed rule called it H-repair.
  - **Worlds.** Built from *this data's* latent fits: per-item posterior draws of (p_U, p_S) and the one-index RL-alone link. Each passes through the identical pipeline, 200 replicates.
    - sym_p: symmetric on the probability scale, q_S = q_R + κ·D;
    - sym_l: symmetric on the logit scale, logit q_S = logit q_R + κ·(logit p_S − logit p_U);
    - in both, κ is common to the two sources and matched by bisection to the mean observed b of the two groups;
    - repair: helped effects kept, hurt effects retained at r, with r matched to the gold group's b.
  - **Rule.**
    - H-repair stands only if d_obs exceeds the 97.5th percentile of d under **both** symmetric worlds.
    - H-sym stands only if d_obs is below the 2.5th percentile of d under the repair world.
    - Otherwise the verdict becomes "intermediate (location-explained)" or "intermediate (repair not excluded)".
  - **Cost and benefit** (`--dry_run power_check`, job 26491169, 60 datasets per world, Part-A sizes, E2-sized effects; fixed rule → fixed rule + check):
    - probability-symmetric worlds: P(H-sym) 0.95–0.97 → unchanged;
    - logit-symmetric world (κ = 0.5): false H-repair 0.13 → 0.02, while P(H-sym) stays 0.60 (the check never upgrades);
    - repair-only: r₀ = 0.45 1.00 → 0.92; r₀ = 0.6 0.82 → 0.50;
    - partial (helped 0.8, hurt 0.5): H-repair 0.87 → 0.37.
    - So the check removes most location-driven false repair calls. In exchange, H-repair needs a clear asymmetry (d ≳ 0.25), and logit-symmetric truths often end as 'intermediate'.
- **Power label at scoring time:** if the fixed rule classifies the sym_p or the repair world correctly < 80% of the time, the verdict is reported as **low-power**.
- **Reported:** the through-origin b±; the undoing and convergence parts of 1 − b±; b± per arm, per dose and per step; marked; logit scale; the probe-only untrained sensitivity.

**A3 — base-success excess** (secondary, per source): the excess defined under Part 0's P0-2, with the same rule. Reported on both scales.

**A4 — history asymmetry: undoing vs convergence** (registered, per source, lenient, step 375):
- Statistic: M − median(symmetric null M), with its hierarchical-bootstrap 95% CI.
- **UNDOING:** CI > 0 and M above the null's 97.5th percentile.
- **CONVERGENCE:** CI < 0 and M below the null's 2.5th percentile.
- **No asymmetry:** otherwise.
- Reported: β_S and β_U with their null values; T's NPMLE free λ_S − λ_R.

**Decision table (Part A → paper → next).**

| Outcome | Paper headline (`paper/reversion` branch) | Next |
|---|---|---|
| H-sym and A1 holds for both sources | BRANCH-SYM: "RL undoes SFT's per-item effects, helpful and harmful alike" | Part X (is it the protocol?) → MATH L3–5 / Countdown dose study; surgery (F6) |
| H-repair (pure or not) | BRANCH-REPAIR: "RL repairs what SFT breaks and keeps what it adds" | Part X; test at n = 50 and on Countdown; allocation corollary (SFT on helpful traces is safe before RL) |
| Intermediate with d ≤ −0.10 | "RL preferentially erases help" (unexpected) | First check the format and teacher-style confound (teacher `## Step` style; marked grader) on the logit scale and per dose |
| Intermediate otherwise | Neutral title; report b± with CIs | Part X merged-fresh arms (X2) decide prospectively |
| A1 fails for both sources | No contraction at 375 steps beyond the null | Report; Part 0 and Part X carry the contraction question |
| A4 UNDOING / CONVERGENCE | Decomposition paragraph: which start moves | — |

## Part B — GSM-Symbolic (exploratory, reported only)
- `14` Part B (5,000 items, k = 1 per adapter; post-RL arms and the untrained model only; no after-SFT samples, so no per-item retention).
- Per family and step: the arm − GRPO-alone pass@1 contrast in points, under lenient, marked, "finished only" (correct and stopped), accuracy among stopped answers, and stop rate.
- Hierarchical bootstrap with template clusters (100 templates; order verified against the HF `original_id`).
- Runs excluded in Part A are excluded here.

## Part X — protocol control: continuing adapter vs merged SFT + fresh LoRA (prospective, ≈ 15–18 GPU-h)
**Why.** Every existing SFT → RL comparison confounds the start with the parameterisation. The SFT start continues its adapter, while RL alone trains a fresh LoRA (B = 0, Kaiming A). Part X crosses the two and gives each protocol its own RL-alone comparator.

**Runs.** 15 GRPO runs in 12 jobs, `unsloth/Llama-3.2-1B-Instruct`, GSM8K, seeds 42–44 (`runs/theory/scratch/submit_prereg15x.sh`; written, not submitted).

| Arm | Start | Protocol | Output (`seed_S/…`) |
|---|---|---|---|
| gold_cont | gold n=250 SFT adapter (`runs/gsm8k/Llama-3.2-1B-Instruct/seed_S/random_sft_250/sft`) | GRPO continues the SFT LoRA | `Llama-3.2-1B-Instruct-px-cont/…/random_sft_250` |
| teacher_cont | teacher n=250 SFT adapter (`…-qwen7b/seed_S/random_sft_250/sft`; `--sft_traces runs/gsm8k/teachers/qwen7b.jsonl` reproduces its SFT/GRPO split) | same | `…-px-cont-qwen7b/…/random_sft_250` |
| gold_mf | gold n=250 | `--merge_sft_start --lora_init_seed S`: SFT LoRA merged into the base weights, fresh LoRA with A drawn by `fresh_lora` after `torch.manual_seed(S)` | `…-px-mf/…/random_sft_250` |
| teacher_mf | teacher n=250 | same | `…-px-mf-qwen7b/…/random_sft_250` |
| RL alone, mf (**MF**) | untrained | `--merge_sft_start --lora_init_seed S` `full_grpo`: the same fresh-LoRA A as the mf arms of seed S | `…-px-mf/…/full_grpo` |
| RL alone, E2 (**E2**, existing) | untrained | E2 `grpo` runs: fresh LoRA drawn by TRL inside `GRPOTrainer`; their `## Step` censor acted on ≤ 3 of 9,600 rollouts | `…-censored-grpo/…/full_grpo` |

- **Settings:** E2's exact command line (W&B run 4w56j5ir, job 25633697), with `--censor` removed.
  - 150 steps; linear decay after 10 warm-up steps; checkpoints every 25 steps;
  - `--reward marked --reward_needs_stop`; `--log_rollouts`; E2's three panel prefixes; `loss_type grpo`;
  - the pipeline's GRPO defaults: 8 prompts × 8 samples, T 0.8, top-p 0.95, 512 tokens, lr 2e-5, KL 0;
  - RL pool: the 750 prompts not used for SFT (1,000 for RL alone);
  - evaluation at T 0.6, k = 4, on all 1,319 test items, for the untrained model, after SFT (`predictions_after_sft.jsonl`, on the merged start for mf) and at step 150 (`predictions.jsonl`).
  - Jobs are ≤ 6 h on `mit_normal_gpu` (E2's two-method job took 1:51–1:58). They are resumable: re-running a line skips finished methods and resumes GRPO from the last checkpoint.
- **Untrained start (p_U):** E2's `base_predictions.jsonl` (12 samples). The Part X jobs' own untrained evaluations reuse the same sampling seeds, so they are probably identical samples; they are not pooled.
- **Comparators.**
  - Primary: cont arms vs **E2**, mf arms vs **MF**. Each protocol is compared with an RL-alone run of the same parameterisation.
  - Secondary (reported): every arm against both RL-alone sets (`<arm>@E2`, `<arm>@MF`).
- **Bootstrap pairing.** The cont and mf arms of a source and seed share the SFT adapter, so they are resampled jointly.

**Predictions** (lenient, probability scale, step 150):
- **X1 (protocol null).** For each source, |b(mf vs MF) − b(cont vs E2)| < 0.15.
  - Reported: the CI of the difference ("equivalent" only if the CI lies inside ±0.15), and the same difference against a common comparator (both arms vs E2; both vs MF).
  - **Kill rule for the headline.** If b(mf vs MF) ≥ 0.85 for **both** sources, the E2/S′ per-item contraction is declared a parameterisation artefact. The paper drops "RL undoes SFT" as a headline, and Part 1 is not run. If it holds for one source, that source's contraction claim is restricted to the continuing-adapter protocol.
- **X2 (help vs damage, prospective).** The A2 rule, with the same thresholds and the same location check, on teacher_mf b₊ and gold_mf b₋ against MF (primary), and on the cont arms against E2 (secondary).
  - With single arms (n = 250 only), the data-adaptive calibration usually returns the **low-power** label (it did in the mock end-to-end test). X2 is therefore read together with A2, not alone.
- **Reported:**
  - the one-index null per arm; A4 per arm; bins; marked; logit scale;
  - stop rates and the training-pool reward gap per arm vs its comparator;
  - gold_cont vs E2's gold `grpo` runs, a same-protocol reproducibility check: the censor never acted on the gold start, so the runs differ only by GPU nondeterminism.

## Part 1 — first-gradient mechanism gate (≈ 7–13 GPU-h; runs only if X1 holds and the kill rule does not fire)
**Starts.**
- The untrained model with a fresh LoRA, and the three gold `mf` starts of Part X. All share the fresh-LoRA geometry and the same A for a given seed, so the untrained-start placebo is clean.
- Teacher `mf` starts are added only if the total stays ≤ 15 GPU-h. They are reported, not registered.
- LoRA init seed = the run's seed (`--lora_init_seed S`). Reward marked + stop, as in training.
- Set C at the untrained start uses the Part X merged-fresh RL-alone finals (`px-mf/…/full_grpo`; same fresh-LoRA A as the mf starts); the E2 untrained finals are a reported alternative.

**Items and samples.**
- 600 GSM8K test items, fixed by `random.Random(15)` over the 1,319 example_ids.
- Disjoint sample sets per start at the RL settings (T 0.8, top-p 0.95, 512 tokens), K = 16 each:
  - A: features;
  - B: pre-RL success;
  - C: post-RL success of the 150-step finals.

**Directions** (all in merged weight space, ΔW summed over adapted layers):
- v₀ˢ is the raw step-0 policy gradient with GRPO advantages (no optimiser step), estimated from 256 prompts of the start's pool × 8 rollouts. At a fresh LoRA its merged form is κ·ΔB·A ∝ G·AᵀA (Prop. 4).
- The random control is drawn in the same LoRA-geometry subspace (rows in the row space of A) and matched on RMS φ (±10%).
- v₁₅₀ is the final merged ΔW; v₂₅ is the merged ΔW to checkpoint-25.
- φ is the directional score, by central finite difference of the sequence log-likelihood. ε is chosen on 32 held-out samples so that the second-difference term is < 10% of the first.

**Input format for the scorer** (`--p1_dir`): `A.jsonl`, `B.jsonl` and `C.jsonl` rows with item, sample, start, direction_name, phi, correct_lenient, correct_marked; optional `covariates.jsonl` (silhouette, length, entropy). Start names are fixed in `score_prereg15.py` (`P1_DOC`).

**Statistics and kill criteria** (R2's corrected gates; lenient primary):
- Δᵛ(x) = mean φ over correct samples − mean φ over incorrect samples (set A). D_SFT = p̂_S − p̂_U (set B).
- Partial Spearman = Spearman of rank residuals after OLS on covariate ranks.
- **K1′.** For each gold-mf start: partial Spearman(Δ^{v₀}, D_SFT | p̂_S) ≤ −0.08, **and** at least 0.05 more negative than both the random-direction statistic (mean over random directions) and the placebo Δ^{v₀} at θ₀ against the same D_SFT. It must hold in ≥ 2 of 3 starts. If K1′ fails, stop the mechanism line.
- **K2′.** Partial Spearman(Δ^{v₀}, q̂_C − p̂_B | p̂_S, p̂_U, D_SFT, f, silhouette, length, entropy), averaged over the 3 gold-mf starts, ≥ 0.10 **and** ≥ 0.05 above the random direction. Uncertainty from a hierarchical bootstrap over starts, then items. If K2′ fails, drop the item-level forecasting claim.
- **K3.** corr over (x, y) of φ along v₀ and along v₁₅₀ ≥ 0.3 at the untrained start **and** at ≥ 2 of 3 gold-mf starts. If K3 fails, relabel the predictor an "early probe" using v₂₅.

**Reported, not predicted:**
- the oracle partial Spearman of the realised change (logit, R2's form) and the share of it captured; all marked versions; per-start values; the base-start forecast at θ₀;
- RMS-φ matching;
- **crossover (exact, paper Prop. 5):** per item, σ̄² = (1 − p)σ₁² + p·σ₀² from set-A φ along v₂₅ (h = 1). The score-function estimate is the more precise iff h·σ̄ < √(2K_SF/K_FD), with K_FD from matched compute K_SF(1 + c_φ) = 2K_FD. c_φ is measured by the tooling. The RMS-log-ratio form is not used.
- GPU-h used.

## Decision summary across parts
| Part | Registered outcome | Consequence |
|---|---|---|
| 0 | P0-1 replicates / does not / intermediate | Does not → no stage-2 reversion study; measurement-paper fallback. Replicates → MATH enters the paper as F1's second task (post hoc label). |
| A | A2: H-sym / H-repair / intermediate / untestable (low-power label if the data-adaptive calibration < 80%) | Selects the paper branch (table above). |
| A | A4: undoing / convergence / none | Decomposition paragraph. |
| X | X1 holds | The protocol is not driving the contraction → Part 1 runs on the `mf` starts; future runs use merged-fresh only. |
| X | X1 fails, no kill | The contraction is protocol-dependent → report both; future runs use merged-fresh only; Part 1 is not run. |
| X | Kill (b(mf) ≥ 0.85, both sources) | The headline contraction is a parameterisation artefact → the paper reframes around parameterisation; Part 1 is not run. |
| X | X2 verdict | The prospective test of A2; if X2 and A2 disagree, the paper reports both and claims neither branch. |
| 1 | K1′ / K2′ / K3 | K1′ fails → mechanism line stops. K2′ fails → no item-level forecasting claim. K3 fails → "early probe" (v₂₅) wording. |

## Scoring commands (after registration; CPU via Slurm; `--unblind` is refused otherwise)
```
cd /orcd/scratch/orcd/006/usemil/data_alloc
S=runs/theory/scratch/score_prereg15.py; O=runs/theory/scratch
sbatch -p mit_normal -c 4 --mem=16G -t 03:00:00 --exclude=node2119,node3006 --output=$O/score_prereg15_part0_out.txt  --wrap "... python -I $S --part 0  --unblind"
sbatch -p mit_normal -c 4 --mem=16G -t 04:00:00 --exclude=node2119,node3006 --output=$O/score_prereg15_partA_out.txt  --wrap "... python -I $S --part A  --unblind"
sbatch ... --output=$O/score_prereg15_partB_out.txt  --wrap "... python -I $S --part B  --unblind"
sbatch ... --output=$O/score_prereg15_partX_out.txt  --wrap "... python -I $S --part X  --unblind"     # after Part X runs finish
sbatch ... --output=$O/score_prereg15_part1_out.txt  --wrap "... python -I $S --part 1  --unblind --p1_dir <dir> --c_phi <measured>"
sbatch ... --output=$O/score_prereg16_out.txt        --wrap "... python -I $S --part 16 --unblind"     # prereg 16
```
`...` is `module load miniforge/25.11.0-0; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /orcd/scratch/orcd/006/usemil/envs/data_alloc; cd /orcd/scratch/orcd/006/usemil/data_alloc;` (as in `runs/theory/scratch/run_p15_dry.sh`).
- Part A is the longest: B = 2000 over 8 groups × 3 steps × 2 graders × 2 scales, plus null fits for 6 arms × 2 graders. Allow 4 h.
- The outcome is appended to this file under "Outcome". Predictions are never edited.

## Dry runs (before registration; CPU Slurm; outputs in `runs/theory/scratch/`)
| Job | What | Output | Result |
|---|---|---|---|
| 26352094 | smoke: compile; tiny power, synthetic and E2 runs | `p15_smoke_out.txt` | runs end to end; E2 reproduced (below); issues found and fixed: logit-scale memory unstable (now probability scale only); `f` NaN under marked; bootstrap choice for selection-based statistics (resolved by the coverage check) |
| 26375736 | schema-only on blinded files | `score_prereg15_schema_out.txt` | 137 adapters, each with the full item set and consistent gold; E2/MATH job files align with probe example_ids (0 mismatches); GSM-Symbolic order verified (100 templates); GSM-Plus 2,093 numeric items, all paired to 300 GSM8K seeds by exact `seed_question`; Part X and prereg-16 outputs absent (not yet run) |
| 26375733, 26376985 (final) | sign-split calibration and power; τ search; registered-rule table | `score_prereg15_dry_power_out.txt` | thresholds above |
| 26376983 | E2 reproduction and full machinery (B = 2000, nsim = 200) | `score_prereg15_dry_e2_out.txt` | below |
| 26376267, 26376984 | planted worlds (recovery), null centring, coverage, prereg-16 power, Part 1 | `score_prereg15_dry_synthetic_s12_out.txt`, `score_prereg15_dry_synthetic_s345_out.txt` | below |
| 26376986 | end-to-end `--part all` on a mock root (real schemas, file names and gold answers; synthetic correctness from a logit-symmetric world, retention 0.5) | `score_prereg15_dry_mock1_out.txt` | every part runs (6 min); exclusion path exercised (an S′ run with stop rate 0.4 dropped); marked regrade 652,100 / 652,100; exposed the location confound (fixed rule: H-repair, d = +0.149) |
| 26432872 | the same, with the location check and the 15-run Part X (both RL-alone comparators) | `score_prereg15_dry_mock_out.txt` | every part runs (6.6 min); the logit-symmetric mock is downgraded from H-repair to 'intermediate (location-explained)': d_obs = +0.149 vs the sym_l world's d [+0.009, +0.189]; Part X X1/X2 run with primary and secondary comparators |
| 26490569 (smoke), 26491169 | power cost of the location check (fixed rule vs fixed rule + check; 60 datasets per world; includes a logit-symmetric world) | `score_prereg15_dry_power_check_out.txt` | see the A2 location-check bullet: no cost in probability-symmetric worlds; false H-repair under logit symmetry 0.13 → 0.02; H-repair at r₀ = 0.6 0.82 → 0.50 |

**E2 reproduction (exact).**
- b = 0.4566 marked and 0.5623 lenient (targets 0.4566 / 0.5623).
- 1 + β_S − β_U equals b exactly.
- R1's 6/6-bin retention is 0.417 (target 0.42).
- The training-pool gap is −0.090 → −0.100 (targets equal).
- Marked regrade = stored field on 94,968 / 94,968 samples; the port equals the live pipeline function on 63,312 / 63,312 pairs.

**E2 under the registered machinery (post hoc; seen before registration; job 26376983).**
These are gold n=250 vs untrained, step 150, hierarchical 95% intervals, null medians [central 95%].
- **Lenient:**
  - b = 0.562 [0.472, 0.669] vs the one-index null 0.983 [0.919, 1.039]. This is the first same-estimator lenient null; the affine null gives 1.027.
  - Logit scale: b = 0.858 [0.738, 1.031] vs its null 1.517.
  - β_S = −0.378 [−0.471, −0.289] and β_U = +0.059 [−0.014, +0.125]: on E2 the contraction is mostly *undoing* at the SFT start.
  - Within the gold arm: b₊ = 0.666 [0.384, 0.951], b₋ = 0.531 [0.444, 0.643]. E2 has no teacher arm, so A2 cannot be scored on it.
  - History asymmetry: M − symmetric-null median = +0.200 [+0.073, +0.333], i.e. undoing toward the base profile.
  - 6/6 bin: retention 0.42 vs a per-bin tilt null of 0.88. The base-success contrast as an excess over the null is +0.06 [−0.36, +0.38]. So R1's "retention falls with base success" is **not** beyond the one-index null on E2.
- **Marked:**
  - b = 0.457 [0.379, 0.539] vs null 0.936.
  - β_S = −0.235 and β_U = +0.309: convergence appears under marked grading. This is the format confound: RL alone learns the answer format.
  - M − null = −0.066 [−0.195, +0.068].

**Synthetic dry runs** (jobs 26376267, 26376984; Part-A sizes; planted truths; estimand = the same statistic on the true per-item probabilities):
1. **Recovery** (100 datasets per world; 8 worlds: symmetric, repair-only, partial, one-index tilt, shared-trait symmetric, attractor-at-base, attractor-at-SFT, explicit memory).
   - The registered estimators (b, β_S, β_U, b±, through-origin, λ_S, λ_R, M) have |bias| ≤ 0.025, and ≤ 0.035 for the secondary through-origin ratios; most are < 0.01 (MC SE ≈ 0.003–0.016). The largest biases sit on the minority side of an arm (gold b₊, teacher b₋), where few items are selected.
   - The naive versions are biased by 0.1–0.35: OLS on the noisy D_pre (attenuation); selecting and measuring on the same samples (regression to the mean); OLS memory regression; β's with shared samples.
   - Single-bin ratios are unstable where D_pre ≈ 0. This is why the base-success contrast uses only bins with |D_pre| ≥ 2 points.
2. **Null calibration** (3 datasets per world, 200 simulations each). In worlds where the null is true (one-index tilt; symmetric shared trait), the observed b, b₊, 6/6-bin retention, logit b and M lie within 1.6 null SDs of the null median. In worlds where it is false (attractor at base, explicit memory), M lies 4.6–10 null SDs away.
3. **Coverage** (30 datasets, B = 200): hierarchical intervals 0.97–1.00 for b, b₊ and b₋; item-only 0.87–0.93.
4. **Prereg 16** (15 datasets, B = 200):
   - Under the null (all b = 0.5), the O1 and O2 rules never fire (0/15 each).
   - O1 with Δb = 0.25 on every OOD set holds with P = 0.80 as designed (CI width ≈ 0.34), and P = 0.93–1.00 with an 8-sample untrained start (width ≈ 0.25).
   - O2 with Δb = 0.15 holds with P = 0.33 in both designs (CI width 0.33–0.45). O2 is underpowered.
5. **Part 1** (600 items, 16 samples per set):
   - Planted world: K1′ holds 3/3 (partial Spearman −0.43 vs random 0.00 and placebo −0.05), K2′ = +0.40 [+0.32, +0.48] vs random +0.02, K3 holds.
   - Null world: K1′ fails 0/3 and K2′ = −0.01.

## Deviations from `prereg_15_draft.md`, with reasons
1. **One-index null = one link fitted on the untrained start.** Start-specific shifts make a two-index model (theory note 3). The affine link is added as a sensitivity check.
2. **Both scales.** The probability-scale null is not 1 (Prop. 1d).
3. **P0-2 restated** as an excess over the per-bin simulated null. Under a tilt, probability-scale retention falls with base success by itself.
4. **Part A added.**
   - The decision rule is calibrated by simulation, and its thresholds differ from the round-4 proposal (H-repair b₊ ≥ 0.85 is unreachable at our noise: full retention reads ≈ 0.78).
   - The sign split uses the Wald chord, not the through-origin ratio, which is not robust to a common offset.
   - A data-adaptive location check guards the verdict against ceiling and floor effects (found in the mock end-to-end test).
5. **Hierarchical bootstrap for every statistic**, chosen by the synthetic coverage check over an item-only alternative.
6. **Untrained GSM8K start pooled with E2's evaluations** (16 samples instead of 4).
7. **Part X added** (lead's change 4). **Part 1 moved to the merged-fresh starts** and made conditional on X1 (lead's changes 5 and 6).
8. **Memory test** (new): EIV projection judged against a symmetric simulated null.

## Appendix — notes on `prereg_16_draft.md` (for the lead)
The scorer implements prereg 16 as drafted (`--part 16`; file names from `submit_prereg16.sh`: `runs/probes/prereg16/{gsm_plus,svamp}_{untrained,sft,s375}.jsonl`, and for MATH `math_untrained.jsonl`, `math_sft.jsonl` + `math_sft_t.jsonl`, `math_s375.jsonl` + `math_s375_t.jsonl`, concatenated per stage).
- GSM-Plus is paired to GSM8K by exact `seed_question`: 2,093 / 2,093 numeric testmini items match 300 seeds; 7 perturbation types.
- The MATH-600 items are keyed by probe example_id.

Statistical problems, and how the scorer handles them:
1. **Untrained start on GSM-Plus has k = 2 (one evaluation), so 1 sample per half.** The EIV denominator stays unbiased but very noisy, and the 25% power gate is itself a noisy estimate. **Recommend** untrained k = 8 on all three sets (≈ 1.5 GPU-h; one adapter). The submission script currently uses k = 2 / 4 / 4. In the synthetic dry run (§4), O1 with Δb = 0.25 holds with P = 0.80 as designed and P = 0.93–1.00 with an 8-sample untrained start (CI width 0.34 → 0.25).
2. **Per-set exclusions break the pairing of O1, O2 and O4.** An adapter excluded on one set but not another changes the seed composition between ID and OOD. The scorer uses **paired composition** as primary (an adapter excluded on any set, including GSM8K, is excluded everywhere) and the draft's per-set rule as a sensitivity check. Register one.
3. **H-uniform** ("|Δb| < 0.10") on a point estimate is not an equivalence claim. Require the 95% CI of Δb inside ±0.10 to state distribution-independence; otherwise "inconclusive". The scorer prints both.
4. **Dependence between GSM8K test and GSM-Plus** (perturbations of 300 test seeds). O1 and O2 need a cluster bootstrap by seed question; the scorer does this. The draft does not say so.
5. **Pooling "at item level"** is the stacked estimator Σ n·Cov / Σ n·Var over sets. It weights sets by items × true variance, so GSM-Plus (2,093 items) will dominate O1 and MATH-600 will contribute little. State it, or register equal set weights. In a first synthetic run (job 26376268; its output file was later overwritten by job 26376984), b = 0.75 on SVAMP and MATH with GSM-Plus at 0.5 moved the pooled Δb by only ≈ 0.10.
6. **The power gate** must be fixed per (set, arm) before the bootstrap, and the ID comparator must use the same arms that enter the OOD pool. The scorer does both; the draft leaves it implicit.
7. **Trivial persistence.** Retention → 1 also happens when RL changes nothing on a set (q_R ≈ p_U and q_S ≈ p_S). O1 would then "hold" without selective erasure. **Recommend** an interpretive guard: O1 is read as "erasure only where RL trains" only on sets where RL alone transfers (q_R − p_U ≥ 2 points, CI > 0); otherwise "OOD persistence is trivial". The scorer reports q_R − p_U per set.
8. **O2's power is low.** The seed side has 300 GSM8K items, and Δ ≥ 0.10 with a CI excluding 0 needs an SE ≈ 0.05. In the synthetic dry run (§4), O2 with Δb = 0.15 holds with P = 0.33 in both designs (CI width 0.33–0.45); under the null it never fires. Either accept O2 as a low-power secondary, or require Δ ≥ 0.25.
9. **The diagnosis rule** "marked vs lenient agreement < 98% on correct-and-stopped answers" needs a direction. The scorer uses P(marked-correct | lenient-correct and stopped).
10. **O3** is a point-estimate ordering with no CI: label it descriptive.

## Outcome, Parts 0, A and B (scored 2026-10-21 with the frozen scorer)
**Jobs and outputs.** 26491530, 26491531 and 26491532 (`runs/theory/scratch/score_prereg15_part{0,A,B}_out.txt`). The working copy was verified byte-identical to `prereg/scripts/p15/`.

**Part 0 — MATH L1–3, post hoc label** (lenient, probability scale, step 375).
- **P0-1 — contraction replicates on MATH.** b = 0.162 [0.072, 0.245] against the one-index null median 0.975.
- **Decomposition.** β_S ≈ −0.8 (undoing at the SFT start) and β_U ≈ 0: the contraction is almost entirely the gold-SFT start reverting.
- **P0-2 not held.** Base-success excess +0.198 [−0.265, +0.641].
- **Pool equalisation (reported).** The training-reward gap goes from −0.181 to −0.053 (ratio 0.29, ≤ 0.3), so the starts equalise on MATH at 375 steps.

**Part A — GSM8K, `08` S′ arms, step 375** (n = 250 and 500 pooled per source).

| Test | Lenient (primary) | Marked |
|---|---|---|
| A1 gold | b = 0.081 [−0.025, 0.185] vs null 0.967: **contraction beyond the null** | b = 0.094 vs null 0.930: holds |
| A1 teacher | b = 0.334 [0.155, 0.543] vs null 0.892: **contraction beyond the null** | b = 0.125 vs null 0.813: holds |
| A2 help vs damage | b₊(teacher) = 0.321 [0.119, 0.696]; b₋(gold) = 0.056 [−0.071, 0.164]; d = +0.265 [+0.042, +0.648] → **intermediate** (asymmetric, but b₊ < 0.50) | d = +0.016 → **H-sym** at registered power |
| A4 gold | M − null = +0.355 [0.215, 0.491]: **UNDOING** toward the base profile | no asymmetry detected |
| A4 teacher | −0.386 [−0.717, −0.073]: **CONVERGENCE** toward the SFT profile | −0.460: CONVERGENCE |
| A3 | gold +0.864 [0.058, 1.871]; teacher not resolved | — |

**Decomposition** (lenient, reported). Retention at steps 75 / 225 / 375:
- gold 0.64 / 0.27 / 0.08;
- teacher 0.66 / 0.46 / 0.33.

The two sources contract by different routes:
- **Gold** is mainly undoing: β_S = −0.71, β_U = +0.21.
- **Teacher** is mainly convergence: β_S = −0.10 (n.s.), β_U = +0.57. Of the helped items' loss of retention, 0.87 is the convergence part and −0.19 the undoing part.

**Part B — GSM-Symbolic** (exploratory). Arm − GRPO alone at step 375, lenient:
- gold-250 +2.2 [+0.1, +4.3];
- teacher-250 +2.4 [−0.1, +4.9].

The marked contrasts are larger (+5.8 / +5.9), partly from format and stop-rate differences. The step-225 contrasts are distorted by a degenerate GRPO-alone run.

**Reading.**
- At 375 steps, RL removes most of SFT's per-item footprint on GSM8K and MATH, whether SFT helped (teacher) or hurt (gold). Damage goes more completely under lenient grading.
- The routes are opposite:
  - **harmful** gold effects are *undone* by the SFT start (toward the base profile);
  - **helpful** teacher effects are *matched* by RL alone (convergence toward the SFT profile).
- Parts X (protocol control) and 1 (first gradient) are pending.

## Outcome, Part X (scored 2026-10-23 with the frozen scorer; job 26664938, `runs/theory/scratch/score_prereg15_partX_out.txt`)
**Runs.** All 15 runs completed. The six still pending were moved from `mit_preemptable`, which was stalled, to `mit_normal_gpu`, with unchanged commands. GRPO stop rates over the last 20 steps were 0.975–0.999, so no run was excluded.

**X1 (protocol null).** Lenient, step 150, b(merged-fresh vs its RL alone) − b(continuing vs E2 RL alone):

| Source | merged-fresh | continuing | Δ | Result |
|---|---|---|---|---|
| gold | 0.647 | 0.653 | −0.006 [−0.161, +0.122] | **HOLDS** |
| teacher | 0.743 | 0.773 | −0.031 [−0.426, +0.417] | **HOLDS** |

- Marked grading: Δ = +0.041 (gold) and +0.051 (teacher); both hold.
- **Kill rule** (b(mf) ≥ 0.85 for both sources): does not fire. The per-item contraction is not a parameterisation artefact. Part 1 may run, on the merged-fresh starts.

**X2 (help vs damage, prospective, step 150).**
- Lenient, merged-fresh (primary): b₊(teacher) 0.618 vs b₋(gold) 0.620, d = −0.002 → **H-sym, LOW-POWER**. The continuing arms (secondary) also give H-sym, low-power.
- Marked: d = −0.462 [−0.654, −0.258] (mf) and −0.431 (cont) → intermediate. Under marked grading the teacher's helped items are erased *more* than gold's damage: b₊ 0.24 vs b₋ 0.70.

**Reading.**
- The protocol does not drive the contraction.
- At 150 steps, retention is 0.65 (gold) and 0.74–0.77 (teacher); the teacher CI reaches 1.1. Contraction deepens with steps (Part A: 375 steps).
- Under marked grading, RL alone learning the answer format accounts for part of the teacher-side convergence.

## Outcome, Part 1 (scored 2026-10-23 with the frozen scorer; `runs/theory/scratch/score_prereg15_part1_i42_out.txt`)
**Runs.** Jobs 26685936–26685943 (`firstgrad.py` via `S2_part1.sbatch`), all completed. Starts are the Part X merged-fresh gold starts (seeds 42–44) and the untrained start, each with that seed's fresh-LoRA initialisation. There are 600 items, and K = 16 samples per set at T 0.8, top-p 0.95.

**Implementation step (format only; mapping fixed before scoring).** `firstgrad.py` writes per-start files. The frozen scorer reads `A/B/C.jsonl` (P1_DOC). The converter `runs/reviews/round4/scratch/lead_p1_convert.py` renames:
- starts: `gold_s{S}_merged` → `gold_mf_s{S}`;
- directions: `v0_lora` → `v0` (the registered first step G·AᵀA), `rand_lora` → `rand_1` (same subspace, matched on RMS φ; matched 1.000), `v25`, `v150`; plus `v0_raw` → `v0raw`, reported only.

It also builds set C (post_gold_mf_s{S}, post_untrained_s{S}) and the registered covariates: per-item length, entropy and silhouette from set-A log-probs.

**The untrained start.** Its first-step directions were computed once per init seed. The scorer has a single untrained start, so seed 42 is primary. Seeds 43 and 44 were scored as sensitivity checks and give the same verdicts (`..._part1_i43_out.txt`, `..._part1_i44_out.txt`).

**c_φ = 0.37.** This is φ along one direction (≈ 55 s per 1k sequences) over generation (≈ 140–165 s per 1k) on L40S (`runs/firstgrad/part1/timings.jsonl`).

**Verdicts.**

| Gate | Lenient | Marked | Held? |
|---|---|---|---|
| K1′ | gold-mf starts passing 0 of 3 | 1 of 3 | **No → the mechanism line stops (registered rule)** |
| K2′ | pooled partial +0.049 [−0.054, +0.148] vs random +0.038 | +0.068 vs +0.046 | **No → no item-level forecasting claim** |
| K3 | corr(φ_v0, φ_v150) = 0.883 untrained; ≥ 0.3 at 2 of 3 gold starts (0.483, 0.710, 0.175) | same | Yes |

**K1′ details** (lenient; partial Spearman with D_SFT given p_S; seeds 42 / 43 / 44):

| Direction | s42 | s43 | s44 |
|---|---|---|---|
| v0 | −0.072 | −0.092 | −0.099 |
| random (same subspace) | +0.034 | −0.082 | −0.060 |
| placebo v0 at θ₀ | +0.046 | +0.047 | +0.042 |

v0 beats the placebo by > 0.05 everywhere, but beats the matched random direction by ≥ 0.05 only at s42, where v0 itself misses the −0.08 bar.

**Reported, not predicted.**
- Realised RL directions anti-align more strongly than v0. v25: −0.173 / −0.144 / −0.115. v150: −0.210 / −0.201 / −0.158.
- Oracle partial (realised change): −0.299 / −0.294 / −0.236; v0 captures 0.24–0.42 of it.
- The untrained start's v0 forecasts RL alone's own per-item change: partial +0.12 / +0.10 / +0.11 lenient, +0.33 to +0.34 marked, every CI above 0.
- Crossover (exact Prop. 5): the score-function estimate is the more precise one on 99–100% of items at the gold starts, and on 10% at the untrained start.

**Reading.**
- The step-0 policy gradient at the SFT start points toward undoing SFT's per-item effects only weakly, and not more than a random direction in the same LoRA subspace. It does not predict *which* items RL reverts.
- The reversion signal appears in the *realised* early updates (v25), which is consistent with OSOL 2609.06469. That observation is post hoc.
- Prereg 17's gradient-based P17-4 and P17-5a, registered as conditional on funding gradient tooling, are therefore not funded and are reported as not run.
