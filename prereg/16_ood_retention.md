# 16 — Does RL erase SFT's per-item footprint only where it trains? Out-of-distribution retention (registered 2026-10-21, before any of its runs)

**Status.** Registered by commit + push to `prereg/` before any prereg-16 evaluation is submitted. The amendments at the end are part of the registration, and they override the design text where the two differ.

**Scorer, frozen at registration:** `prereg/scripts/p16/score_prereg15.py --part 16`, with `p15lib.py` and `p15_mock.py`. SHA-256:
- `score_prereg15.py` 668a14a57c68a83e0b605f2ef2733a77ba2b22267b0c4e720d6e3acfc3c04dab
- `p15lib.py` 9c15684cbdd0ec13e00f6cf36034b475ad51fb22a2b16dae97413735606edf0d
- `p15_mock.py` 5fbdbe5f22b425ddafbe6748aa4de6925e1fe2f1972d0bc417be7d56166bca87

**Submission script:** `prereg/scripts/p16/submit_prereg16.sh` (00b15b9786a2e43309c6e3d30a4537e79490ba871fd84b8160b17b5ca6d499fb).

**Dry runs** were synthetic only (`runs/theory/scratch/score_prereg15_dry_synthetic_s345_out.txt` §4):
- under the null, O1/O2 fire 0/15;
- O1 power is 0.80 at Δb = 0.25, and 0.93–1.00 with an 8-sample untrained start;
- O2 power is 0.33.

**Additionally seen by the lead before registration** (aggregate GSM-Symbolic pass@1 contrasts computed by the prereg-14 session, relayed on 2026-10-19). Arm − GRPO alone at 375 steps:
- gold-250: +5.8 on GSM-Symbolic vs +1.0 on GSM8K;
- teacher-250 / 500: +5.9 / +7.2 vs +2.8 / +3.5.

GSM-Symbolic is not one of this registration's sets. The aggregate gaps motivated O4, which is therefore not a blind prediction for direction; its threshold was set without seeing GSM-Plus, SVAMP or MATH.

**Hypothesis.**
- On held-out items close to the RL training distribution, GRPO pulls an SFT start's per-item success profile toward the RL-alone profile (retention well below 1; GSM8K E2: 0.56 lenient).
- On items further from that distribution, more of SFT's per-item effect survives.
- This is the corollary of the linearised kernel account (`paper/reversion` Prop. 6; Doan et al. 2010.04003; Hidekel & Raviv 2606.18024): the post-RL difference is D_post(x) = D_pre(x) − k(x, X)(K + λI)⁻¹D_pre(X), and k(x, X) → 0 for items weakly aligned with the RL pool X.
- **What was seen before registration.** Another session saw aggregate GSM-Symbolic contrasts at 375 steps: SFT arms beat GRPO alone by more there than on GSM8K. Those numbers are unscored and not part of this registration.
- **What has not been seen.** No one has evaluated these adapters on GSM-Plus, SVAMP or MATH. The sets below are therefore prospective.

## Design

**Adapters** (all exist; Llama-3.2-1B-Instruct, GSM8K pool, `08` S′ runs, 375 steps, marked reward):
- the untrained model;
- GRPO alone at step 375, seeds 42–44: `runs/gsm8k/Llama-3.2-1B-Instruct-marked-steps375/seed_S/full_grpo/grpo/checkpoint-375`;
- for gold and teacher (Qwen2.5-7B-I) traces at n ∈ {50, 250, 500}, seeds 42–44:
  - the after-SFT adapter: `runs/gsm8k/Llama-3.2-1B-Instruct[-qwen7b]/seed_S/random_sft_n/sft`;
  - its step-375 checkpoint: `…-marked-steps375[-qwen7b]/seed_S/random_sft_n/grpo/checkpoint-375`.

40 adapters in total.

**Evaluation sets** (`probe.py --dataset …`, `--reward marked`, T 0.6, top-p 0.95; graded lenient (primary) and marked):

| Set | Items | Samples / item / adapter | Max tokens | Distance from the RL pool (GSM8K train) |
|---|---|---|---|---|
| GSM-Plus testmini (numeric; perturbations of GSM8K **test** questions) | all numeric items (loader in `route_sft_grpo.py`) | 2 | 512 | near: same problems, perturbed |
| SVAMP (train + test, 1,000) | 1,000 | 4 | 512 | medium: different elementary word problems |
| MATH levels 1–3 numeric test, the fixed 600-item subset that `probe.py --n_prompts 600` selects (`random.Random(0)` on the `--data_seed 1234` test order) | 600 | 4 | 1,024 | far: different domain |

The GSM8K test set (in-distribution, ID) comes from `14` Part A (same adapters, k = 4, T 0.6).

**Cost.** Estimated with measured throughputs from `14` (20k GSM-like samples/GPU-h; 7.9k MATH samples/GPU-h): ≈ 8 + 8 + 12 ≈ 28 GPU-h on `mit_normal_gpu`.

## Statistics
`runs/theory/scratch/score_prereg15.py --part 16`, the same estimators as `15` Part A.

**Per set and per arm:**
- p_U, p_S (pre-RL) and q_S, q_R (step 375).
- D_pre = p_S − p_U and D_post = q_S − q_R.
- Retention b is the EIV slope of D_post on D_pre (G1's `rho_item`).
- Sign-split b₊ / b₋ use disjoint halves.

**Pooling.** "Pooled" means gold n ∈ {250, 500} and teacher n ∈ {250, 500}, each source pooled over doses. The hierarchical bootstrap resamples seeds within arm, then items (B = 2000).

**Power gate.** A set is scorable for an arm only if the true pre-RL variance Var_true(D_pre) is ≥ 25% of that arm's GSM8K value. Otherwise it is "untestable" for that arm.

**GSM-Plus pairing.** Each perturbed item maps to its seed GSM8K test question by exact question text, using the dataset's seed-question field.

## Registered predictions (lenient primary)
- **O1 (primary).** For each source, the pooled OOD retention (GSM-Plus, SVAMP and MATH pooled at item level) exceeds GSM8K-test retention by ≥ 0.15, with the 95% CI of the difference excluding 0.
- **O2 (paired).** On GSM-Plus perturbed items vs their GSM8K seed questions (the same seeds only), retention on the perturbed items exceeds retention on the seeds by ≥ 0.10, with the CI excluding 0. Teacher and gold are pooled.
- **O3 (ordering; secondary).** b_MATH ≥ b_SVAMP and b_MATH ≥ b_GSM-Plus for each source; a point-estimate ordering.
- **O4 (aggregate).** For teacher n ∈ {250, 500} pooled, the step-375 gap (SFT arm − GRPO alone) averaged over the three OOD sets exceeds the GSM8K-test gap by ≥ 1.0 point. This uses sampled pass@1 with its hierarchical CI; the claim is directional and its CI must exclude 0.

**Competing account (H-uniform).** Reversion does not depend on distance: |b_OOD − b_ID| < 0.10. If O1 fails with |Δb| < 0.10 for both sources, the paper drops the OOD-persistence claim.

## Decision table

| Outcome | Interpretation | Next step |
|---|---|---|
| O1 holds for both sources | RL erases SFT's per-item footprint mainly where it trains | Headline candidate. Run the kernel-alignment test (`17`): retention vs measured k(x, X) on Countdown/MATH L3–5 |
| O1 holds for one source | Source-dependent | Report it. Check trace style/length as the confound |
| O1 fails, \|Δb\| < 0.10 | Reversion is distribution-independent | Drop OOD persistence. The paper leans on Part A of `15` |
| b_OOD < b_ID by ≥ 0.15 | RL erases SFT effects *more* off-distribution | Surprising. Check for format failure on OOD sets before interpreting |

## Reported, not predicted
- RL-alone transfer per set: q_R − p_U.
- Stop rates and "finished-only" grading per set: the failure-mode check.
- Marked-grading versions of everything.
- Per-dose (n = 50) results.
- The per-perturbation-type breakdown on GSM-Plus.

**Exclusions.**
- An adapter whose stop rate is < 0.5 on a set is reported and excluded from that set.
- If, after exclusion, an arm has < 2 seeds on a set, that set is untestable for the arm.

## Diagnosis rules (before interpreting any failure)
- **Grader bug:** marked vs lenient agreement < 98% on correct-and-stopped answers. Hand-check 50 items and fix the grader.
- **Format or truncation:** stop rate < 0.9 for an arm on a set.
- **Contamination:** for SVAMP and GSM-Plus, the CoDeC rows in `runs/reviews/round3/scratch/t2_cont_out/`.
- **Null degeneracy:** too little SFT effect variance on a set; this is the power gate above.

## Amendments before registration (lead, 2026-10-20; adopting the scorer engineer's notes in `prereg_15_final_draft.md` appendix)
1. **Untrained start sampled at k = 8** on all three sets (one adapter, ≈ 1.5 GPU-h). This replaces k = 2 / 4 / 4, which left one sample per half on GSM-Plus.
2. **Paired composition (primary).** An adapter excluded on any set, GSM8K test included, is excluded everywhere. The per-set exclusion rule is reported as a sensitivity check.
3. **H-uniform** ("distribution-independent") is claimed only if the 95% CI of Δb lies inside ±0.10. A point estimate inside ±0.10 with a wider CI is "inconclusive".
4. **Cluster bootstrap by GSM8K seed question** for every comparison involving GSM-Plus: 2,093 numeric testmini items, 300 seeds, 7 perturbation types.
5. **Pooling.** The primary O1 statistic is the stacked estimator Σ n·Cov / Σ n·Var over the OOD sets. It is item- and variance-weighted, so GSM-Plus dominates. The equal-set-weight average of per-set (b_OOD − b_ID) is secondary.
6. **Power gate fixed per (set, arm) before the bootstrap.** The ID comparator uses exactly the arms that enter the OOD pool.
7. **Trivial-persistence guard.** O1 is read as "erasure mainly where RL trains" only on sets where RL alone transfers: q_R − p_U ≥ 2 points with a CI > 0. On other sets, high retention is reported as "trivial persistence (RL changes little there)".
8. **O2** is registered but labelled underpowered: P ≈ 0.33 at Δb = 0.15 in the synthetic dry run.
9. **O3** is descriptive (a point-estimate ordering).
10. **Diagnosis rule direction:** P(marked-correct | lenient-correct and stopped) < 0.98 triggers a grader check.
