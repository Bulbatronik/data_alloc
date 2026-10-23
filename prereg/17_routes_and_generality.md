# 17 — Routes and clock of what RL undoes of SFT: new GSM8K arms (17A) and a MATH L3–5 two-family generality test (17B) (registered 2026-10-23, before any 17A/17B run)

**Status.** Registered by commit + push to `prereg/`. The draft text below is the registration; its "lead" placeholders are resolved in place.

**Scripts, frozen at registration** in `prereg/scripts/p17/` (SHA-256):
- `p15lib.py` 9c15684cbdd0ec13e00f6cf36034b475ad51fb22a2b16dae97413735606edf0d
- `p17_make_math_ids.py` 7409118d68e0acc513136c037ef093370e78672265fef0aba404a62fd5569876
- `p17_make_tunc_ids.py` 30541443863bb2f3e9c1d1ef31d8e4dc5d7d79e9dab0c94bc76ea1f0071c0bb2
- `score_prereg15.py` 9def2afa975ecfca4ad8769f23fdc6cf7c82ca275da4da5ab21be6b48ee32e31
- `score_prereg17.py` 82743ff54c9c21afcfd23932aec609569e622b31b1e754136e4fb040b3e2052f
- `submit_prereg17a.sh` 3d28054b37e837caf89bbd6df6cb807cd302373a72382e429e42ff121aabd944
- `submit_prereg17b.sh` f6154d2e287a2bf730db78631744693ef8c85eec6d0eed09610097e5005f7761

**Seen by the lead before registration, in addition to the draft's disclosures:**
- prereg 15 Part X outcome: the protocol null holds, no kill; X2 is H-sym, low-power, under lenient grading, and intermediate (d = −0.46) under marked grading;
- prereg 16 outcome: off-distribution retention ≈ 0.1, so OOD persistence is not supported.

Neither changes any 17 threshold, which was fixed by simulation. 17B uses `PROTO=mf` because X1 held.

**Partitions.** 17A runs on `mit_preemptable` with requeue (resumable chains); 17B runs on `mit_normal_gpu`. The commands are otherwise as frozen.

## Context
- **What 15 found.** On GSM8K (08 S′, 375 steps) and MATH L1–3, RL removes most of SFT's per-item footprint relative to RL alone. The routes differ:
  - harmful (gold) effects are **undone** by the SFT start;
  - helpful (teacher) effects are **matched** by RL alone (convergence).
- **The attractor model with partial reachability** (AR; `paper/reversion` §4.1, Def. 2, Props 8–11): RL relaxes centred per-item logit profiles toward one start-independent attractor on a reachable subspace.
  - **Identities, not evidence:** b = 1 + β_S − β_U; undoing u and convergence c with u + c = 1, where c = β_U / (1 − b); β_U computed from RL alone.
  - **AR's falsifiable content:** a common clock (cumulative learning rate, single rate); a head start (RFT, and SFT on RL alone's own samples) converges and does not undo (Prop. 11); arms are ordered by their unreachable share ρ⊥; no amplification.
- **Counter-examples and how 17 answers them** (`runs/reviews/round4/L2_scoop_watch.md`):
  - **Li 2609.37104:** SFT accuracy damage survives correctness-only RL (no RL-alone arm, one budget). 17B asks on a non-saturated standard benchmark (MATH L3–5), with an RL-alone arm and two checkpoints, whether gold damage is undone there too.
  - **Dong 2609.28145 and Aphale 2606.18487:** no convergence at their budgets. 17A's clock test and 17B's two checkpoints report retention as a function of RL time, not one budget.
  - **Kim 2505.14216:** teacher help not reachable by RL. T_unc (teacher traces only on items the base model never solves) and the 17B teacher arm test whether teacher help is matched by RL alone when the base model cannot sample it.
  - **Lu 2607.25063, Duan 2610.04978:** the destination depends on the base (pretraining) model. 17B's second student family (Gemma-3-1B-it) tests whether the route pattern holds across bases.

## Common conventions (as `prereg/15`, plus the quantities below)
- **Inherited from 15:**
  - Lenient (format-insensitive) grading primary, marked secondary.
  - Probability scale primary; smoothed logit reported.
  - Halves = samples < k/2.
  - EIV retention b (G1 `rho_item`); cross-fitted β_S and β_U.
  - Hierarchical bootstrap: seeds within lineage, then items; B = 2000.
  - One-index (tilt) null: one link fitted on the untrained start's own pair, 200 replicates through the identical pipeline.
  - Exclusion: a run with stop rate < 0.5 over its last 20 steps; a cell with < 2 seeds is not scored.
- **Convergence share** c = β_U / (1 − b), defined only if 1 − b ≥ 0.15; undoing share u = 1 − c. These are identities, so routes are reported but only *the predicted values of c for new arms* are tests.
- **Level retention** = mean post-RL logit gap / mean pre-RL logit gap (smoothed logits). Defined only if |pre gap| ≥ 0.10.
- **Clock** τ(t) = cumulative learning-rate factor over steps 0 … t−1 (HF schedules with 10 warm-up steps):
  - constant: τ = 69.5 / 144.5 / 219.5 / 294.5 / 369.5 at steps 75 / 150 / 225 / 300 / 375;
  - linear decay over 375 steps: τ = 63.8 / 117.8 / 156.5 / 179.7 / 187.5.
- **Power gate:** a route or level statistic is "untestable" for an arm if Var_true(D_pre) < 0.004 or 1 − b < 0.15.

---

## 17A — GSM8K, the 08 S′ protocol, new arms (≈ 53 GPU-h)

**Protocol.** Identical to 08 S′ (W&B run l6fbde5d), so the existing S′ gold n=250, teacher n=250 and GRPO-alone arms are the reference:
- Llama-3.2-1B-Instruct, LoRA, continuing adapters;
- 375 GRPO steps; constant lr after 10 warm-up steps; marked reward;
- checkpoints every 75 steps; `--log_rollouts` added (measurement only).

| Arm | SFT data (n = 250 SFT prompts of the seed's partition unless stated) | RL |
|---|---|---|
| **RFT** | the untrained model's own **marked-correct** samples (pool, k = 8, T 0.8, top-p 0.95; first correct per prompt; prompts without one are dropped, n reported). The old `traces/self.jsonl` (k = 4, graded without the marked grader) is not reused | the seed's 750 GRPO prompts, constant lr |
| **HS** (head start) | own marked-correct samples of the S′ GRPO-alone **checkpoint-75** of the same seed (same sampling) | same |
| **T_unc** | teacher (Qwen2.5-7B-I) traces on pool prompts with **lenient pass@8 = 0** for the untrained model (`p17_make_tunc_ids.py`); up to 250 by `random.Random(17)`, else all (n reported) | the rest of the 1,000-prompt pool (`listed_sft`), constant lr |
| **gold-linear** | the S′ gold n=250 adapters, reused | 750 prompts, **linear decay over 375 steps** |
| **RL alone, linear** | — | 1,000 prompts, linear decay (gold-linear's comparator) |

- Seeds 42–44: 15 runs.
- **Evaluation:** `probe.py`, T 0.6, k = 4, all 1,319 test items, after SFT and at steps 75 / 150 / 225 / 300 / 375.
  - The S′ checkpoints 150 and 300 of GRPO alone, and checkpoint 150 of gold n=250, are evaluated too, as comparator and clock reference.
  - The untrained start is prereg 15's: probe 4 samples + E2 12 samples.
- **Comparators:** RFT, HS and T_unc against S′ GRPO alone; gold-linear against RL alone linear.

**Registered predictions** (lenient, probability scale, step 375 unless stated). The thresholds are calibrated by simulation: `--dry_run synthetic`, job 26665000, 60 datasets per world at 17A sizes, worlds built from AR with S′-like parameters (synthetic gold b(375) 0.08, teacher 0.31, gold-linear 0.26).

| ID | Prediction (W's form, recalibrated where marked) | P(held \| true) | P(held \| false) |
|---|---|---|---|
| **P17-1b clock** | gold-linear b(375) is within 0.10 of S′ gold-250 interpolated at the same cumulative lr (τ 187.5, between the S′ checkpoints 150 and 225), **and** exceeds S′ gold-250's step-matched b(375) by ≥ 0.10 | 0.95 | 0.00 (step clock) |
| **P17-1a early probe** | gold-linear: \|b(375) − b(75)^{τ(375)/τ(75)}\| ≤ 0.12 | 0.92 | 0.17 (step clock) |
| **P17-2 RFT** | c ≥ 0.6 and β_S ≥ −0.25. *Recalibrated:* W's c ≥ 0.8 and β_S ≥ −0.10 held in only 0.48 of datasets when RFT is truly convergence-only, because β_S has SD 0.13 | 0.85 | 0.00 (RFT undoes) |
| **P17-2 HS** | c ≥ 0.9 and β_S ≥ −0.05 (W's values; HS's β_S is far above 0 under AR) | 1.00 | 0.00 |
| **P17-2 falsification** | RFT or HS shows β_S ≤ −0.30 with the 95% CI entirely below −0.10. The head-start reading of convergence is then wrong | — | point part 0.98–1.00 |
| **P17-5b** | T_unc has the largest b(375) of all SFT arms (S′ gold, S′ teacher, RFT, HS), **and** b(T_unc) − b(teacher-250) has a 95% CI above 0 | 1.00 (point) | not simulated |
| **P17-5c** | RFT and HS b(375) ≤ 0.25. *Recalibrated:* W's 0.15 held in 0.68 of datasets when true (b SD 0.08 and 0.15) | 0.92 | — (needs ρ⊥ > 0 for RFT/HS) |
| **P17-5d** | Coverage does not order the arms: \|Spearman(unsolved share, b(375))\| < 0.5 across the 5 SFT arms. **Low power:** 5 arms, simulated Spearman SD 0.48 | ≈ 0.6 | — |
| **P17-6 bounds** (corrected) | For each new arm, probability-scale b does not rise by > 0.10 between consecutive checkpoints with a bootstrap CI above 0, and the logit-scale b(375) does not exceed its own one-index null median by > 0.10 | — | — |
| **P17-7 level** | RFT and HS level retention ≤ 0.3 (untestable if \|pre gap\| < 0.10 logit) | high (simulated 0.06 ± 0.12) | — |

- **Why P17-6 is corrected.** Smoothed-logit b is biased upward with few samples: on E2 the one-index null itself reads 1.5. W's raw "logit b in [−0.10, 1.10]" would fail under no reversion at all.
- **Not registered, reported only:**
  - **P17-3 semigroup** (HS's post-RL difference vs RL alone's own increment q_R(t + 75) − q_R(t)). It is not identifiable at these sizes. RL alone's increments are nearly flat after t = 75, so the EIV denominator vanishes. In the simulated world where it is true, the pooled t = 75/150 slope has SD 6.8; it "held" 0.48 and was "falsified" 0.28. Prop. 11's content is still tested by P17-2 HS (β_S ≥ 0, c ≥ 1).
  - **P17-4 (pre-RL route forecast from step-0 gradients) and P17-5a (kernel-alignment ρ̂⊥).** They need the first-gradient and gradient-sketch tooling (`firstgrad.py`) on 6 starts (≈ 10–15 GPU-h, not budgeted). They are registered here so that they are prospective if the lead funds them:
    - P17-4: Spearman(c(0+), c(375)) ≥ 0.8 across the 5 SFT arms, and the same side of 0.5 for ≥ 4 of 5;
    - P17-5a: Spearman(ρ̂⊥, b(375)) ≥ 0.8.
    - If not run: "not scored".
  - Per-arm b, β, c, level and M at every checkpoint, on both scales and both graders; the n of each arm; RFT/HS/T_unc trace lengths.

**Decision table (17A).**

| Outcome | Reading | Next |
|---|---|---|
| P17-1b held | RL time is cumulative lr: retention transfers across schedules; budgets must be compared on τ, not steps | Paper: clock claim (with E2 retrodiction as supporting, post hoc) |
| P17-1b not held, step-matched gap < 0.10 | The step count, not lr, is the clock (or schedule matters otherwise) | Drop the clock claim; report retention per step only |
| P17-2 held for RFT and HS | Convergence is a head start: SFT on the policy's own correct samples is undone by nothing and matched by RL alone | Paper: head-start reading of the teacher route (BRANCH-ROUTES) |
| P17-2 falsified (RFT or HS undo) | The shared-attractor account is wrong: even own-sample SFT is undone | AR is descriptive only; drop Prop. 11 claims |
| P17-5b held | Teacher help on items the base cannot solve survives RL (ρ⊥ large): reachability, not teacher identity, sets retention | Paper: "RL removes what it can reach" |
| P17-5b not held | Help on unsolved items is also matched or undone; no reachability evidence | Report; 17B and prereg 16 carry the reachability question |
| P17-5d not held | Coverage orders the arms: item-wise reachability is revived (contrary to W's post hoc reading) | Report |

---

## 17B — MATH L3–5, two student families (≈ 155 GPU-h)

**Data** (`data/math35/`, built by the dataset engineer; `route_sft_grpo.py --dataset math35d`):
- fixed SFT pool of 1,500 items, RL pool of 1,000, test set of 1,000 (367 MATH-500 L3–5 + 633 other L3–5 test items);
- prompt `boxed_user`; graded by the pipeline's sympy `math35d` grader, lenient and marked.

**Students:** Llama-3.2-1B-Instruct and Gemma-3-1B-it.

**Arms**, per family:

| Arm | SFT | Seeds |
|---|---|---|
| RL alone | — | 42–44 |
| gold | Hendrycks solutions (`--sft_traces gold`) | 42–43 |
| teacher | Llama-3.1-8B-Instruct traces, marked-correct (`data/math35/traces/teacher_llama8b.jsonl`) | 42–43 |
| RFT | the student's own marked-correct samples (`data/math35/traces/{llama1b,gemma1b}.jsonl`) | 42–43 |

- **SFT items:** n = 500 per arm (`p17_make_math_ids.py`). The 1,500 SFT-pool ids are ordered by `random.Random(17)`, and each arm takes the first 500 that have its trace type. If coverage is lower, it takes all covered items and n is reported, as are the overlaps.
- **RL:** on the RL pool only. 300 GRPO steps; 1,024 tokens; marked + stop reward; constant lr after 10 warm-up steps; micro-batch 4 × 16 (as prereg-14 Part C); checkpoints at 150 and 300; `--log_rollouts`.
- **Protocol placeholder** (`submit_prereg17b.sh`, `PROTO`):
  - **mf**: merged SFT + fresh LoRA, `--merge_sft_start --lora_init_seed S` for every arm including RL alone; used if Part X's X1 holds;
  - **cont**: continuing adapter; used otherwise.
  - **Fixed at registration: `mf`.** Part X X1 held for both sources (prereg/15 Outcome, commit ddf191f). The scorer takes `--proto mf`.
- **Evaluation:** `probe.py`, T 0.6, top-p 0.95, 1,024 tokens, all 1,000 test items:
  - untrained k = 24;
  - after SFT, checkpoint-150 and checkpoint-300 at k = 8.
  - The training jobs' own evaluations are greedy on 100 items and are not scored.

**Prerequisites (blocking).**
1. The dataset engineer's trace files `data/math35/traces/{teacher_llama8b,llama1b,gemma1b}.jsonl`. They are being generated; jobs pending as of 2026-10-23.
2. A pipeline option `--grpo_ids <json>` that restricts the GRPO bucket to the listed ids.
   - `math35d`'s training split is the SFT pool followed by the RL pool. Without the option, `full_grpo` trains RL alone on all 2,500 items, and `listed_sft` with a 500-item subset puts the remaining 1,000 SFT-pool items into the RL bucket.
   - The option does not exist yet. The submission script refuses to run without it.
   - **Resolved before registration:** `--grpo_ids` was added with regression tests (commit 89fcb08), and all three trace files exist (coverage in the lead amendments below).

**Registered predictions**, per family and grader (lenient primary), step 300:

| ID | Prediction | Simulated power (17B sizes, one family, job 26665000) |
|---|---|---|
| **B1 contraction** | gold and teacher each: b < median(one-index null b) − 0.10, with the 95% CI also below that value | ≈ 1 when b ≈ 0.1–0.2 against a null ≈ 0.9 (not simulated with nulls; the bar is far from the simulated values) |
| **B2 routes** | c_gold ≤ 0.35 (undoing-dominated), c_teacher ≥ 0.65 and c_RFT ≥ 0.65 (convergence-dominated); **and** the ordinal c_gold < c_teacher with a 95% CI of c_teacher − c_gold above 0, and c_teacher ≤ c_RFT + 0.15 | 0.97–1.00 when the routes are as predicted (point parts); 0.87 for the order part when teacher = RFT; **0.00** in a world with no route difference (all c = 0.5). The margins replace a 0.5 split, which gave 0.63 false positives |
| **B3 level** | level retention teacher − gold > 0 with a 95% CI above 0 | point part 1.00 when true, 0.48 when there is no level difference (the CI requirement removes those) |

- **Generality** = B1, B2 and B3 held in **both** families. In one family only: "family-dependent".
- **Reported:** steps 150 and 300; b, β, c and level per arm; M; MATH-500 vs the other test items separately; marked versions; coverage and n per arm; stop and format rates.

**Decision table (17B).**

| Outcome | Reading | Next |
|---|---|---|
| B1–B3 hold in both families | The routes generalise: on a hard, non-saturated benchmark, RL undoes harmful SFT and matches helpful SFT, across bases | Paper headline (BRANCH-ROUTES), with Li et al. addressed (RL-alone-referenced, per item, two budgets) |
| B1 fails for gold | Damage survives RL on MATH L3–5 at 300 steps (Li-like) | Report as a regime boundary; check whether retention falls between 150 and 300 (budget) |
| B2 fails: teacher undone (c_teacher < 0.65) | Teacher help is not matched by RL alone when the task is hard (Kim-like) | Report; T_unc (17A) and prereg 16 frame it as reachability |
| Holds for one family only | Base-dependent (Lu, Duan) | Report as family-dependent; no general claim |

**Cost (unverified; replace with the dataset engineer's measured timing, S3 smoke job 26526538).**
- Training: 18 runs; ≈ 3.9 h per 375 steps on MATH L1–3 (prereg-14 Part C), × 0.8 for 300 steps, × 1.3 for L3–5 lengths, × 1.3 for Gemma → ≈ 81 GPU-h.
- Evaluations: 2 × 216k samples at ≈ 6k per GPU-h → ≈ 72 GPU-h.
- Total ≈ 155 GPU-h: about 3.3 days wall-clock at the 2-GPU `mit_normal_gpu` limit.
- Cheaper option: k = 4 at checkpoint-150 saves ≈ 12 GPU-h.

---

## Scoring and submission
- **Submission** (mit_normal_gpu, ≤ 6 h per job, `--exclude=node2119,node3006`):
  - Every training and evaluation step is a chain of jobs linked by `afterany`. Training resumes from the last checkpoint; `probe.py` skips adapters already written. Downstream steps wait on `afterok` of the last job of each upstream chain.
  - `bash runs/theory/scratch/submit_prereg17a.sh`
  - `PROTO=mf|cont bash runs/theory/scratch/submit_prereg17b.sh`
- **Scoring** (CPU via Slurm; `--unblind` is refused otherwise):
  - `sbatch -p mit_normal -c 4 --mem=16G -t 04:00:00 runs/theory/scratch/run_p17_dry.sh --part 17A --unblind`
  - `sbatch … --part 17B --proto <PROTO> --unblind`
- **Schema checks** (no outcome field read): `--part {17A,17B} --schema_only`.

## Dry runs (CPU Slurm; outputs in `runs/theory/scratch/`)
| Job | What | Output | Result |
|---|---|---|---|
| 26526926 | first synthetic smoke | `p17_smoke_out.txt` (overwritten) | failed: a generator bug (gold-linear spec), fixed |
| 26527008 | synthetic smoke, 2 datasets per world | `p17_smoke_out.txt` | runs; showed that W's RFT thresholds and the HS semigroup at t = 150/300 do not behave |
| 26527093 | synthetic, 60 per world, before recalibration | (overwritten by 26665000) | P17-2 RFT 0.48 when true; B2 split 0.63 false positives |
| 26665000 | synthetic, 60 per world, registered thresholds | `score_prereg17_dry_synthetic_out.txt` | the power column above |
| 26665001 | schema-only, 17A and 17B | `score_prereg17_schema_out.txt` | all inputs correctly reported missing (nothing run yet) |

- **Not yet exercised end to end:** the 17A and 17B data loaders. No 17 data exist.
  - 17A reuses prereg 15's tested loaders plus a probe-file parser.
  - 17B regrades with the pipeline's `math35d` grader. It must reproduce the stored marked `correct` on the first files (agreement is printed).
  - **[LEAD: run a mock end to end once the first 17 files exist, as for 15.]**

## Deviations from W's recommendations (`W_attractor_note.md` §8), with reasons
1. **Arms and protocol follow the lead's 17A design.**
   - New arms only, under the 08 S′ protocol (continuing adapters), so the S′ arms are the reference; W suggested merged-fresh if Part X shows protocol dependence (17B carries that choice).
   - **Added an RL-alone linear-schedule arm**, which the clock test needs as gold-linear's comparator; without it, the comparator would assume the clock under test.
   - The linear schedule spans 375 steps (τ(375) = 187.5), not 150 steps (that would be E2 again).
2. **P17-2 RFT thresholds recalibrated** (0.6 / −0.25 instead of 0.8 / −0.10). HS keeps W's.
3. **P17-3 demoted to reported** (not identifiable at these sizes).
4. **P17-5c bar 0.25** instead of 0.15 (noise).
5. **P17-6 restated against the logit-scale null** (smoothing bias).
6. **P17-4 and P17-5a registered but conditional** on unbudgeted gradient tooling.
7. **T_unc uses lenient unsolved.** A marked criterion would count format failures as unsolved.

## Lead amendments before registration (2026-10-23)
1. **17B design inputs** (dataset engineer, job 26684486; untrained, T 0.8, k 8, 1,024 tokens). These were seen before registration. Marked pass@1 / ≥1-of-8 / stop rate:

   | Model | Test | RL pool |
   |---|---|---|
   | Llama-3.2-1B-I | 0.109 / 0.359 / 0.851 | 0.206 / 0.572 / 0.874 |
   | Gemma-3-1B-it | 0.335 / 0.504 / 0.525 | 0.336 / 0.523 / 0.530 |

   - **Llama-1B solves MATH-*train* items about twice as often as test items; Gemma does not.** Together with Llama-1B-I's ambiguous CoDeC of 0.65 on MATH-train problem text (`D_datasets.md` §4), this is a possible post-training exposure. **Gemma-3-1B-it is therefore the clean family for 17B.** Every 17B claim must hold in Gemma. Llama results are reported with this caveat.
2. **Gemma stop rate is 0.53 at 1,024 tokens**, from verbosity, close to the 0.5 exclusion bar. The exclusion rule is unchanged: stop rate < 0.5 over the last 20 RL steps. Gemma's early RL gains will partly be learning to stop. Both graders and "finished-only" accuracy are reported.
3. **Submission-script fixes:**
   - `--max_train_samples -1`, so all 2,500 items load and the fixed RL pool is complete;
   - `--sft_max_length 2048`, because 1,024 truncated 2.8–9.2% of SFT rows;
   - probe batch 256.
4. **Trace coverage** on the 1,500-item SFT pool (first marked-correct): teacher 75.4%; RFT Llama 54.3%; RFT Gemma 49.7%. All arms reach n = 500.
