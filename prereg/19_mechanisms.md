# 19 — Why RL erases SFT's per-item effect: mechanism tests

**Part I registered 2026-11-04, before any Part I run was launched.** Part II (the analysis-only mechanisms M1–M4) will be appended as a dated section before any of its statistics is computed.

**Motivation.** Prereg 15–18 established a description:
- RL (GRPO, no KL term) erases most of SFT's per-item effect, with retention b ≈ 0–0.33 against a one-index null ≈ 0.9–1.0;
- the effect disappears by undoing (harmful arms) or matching (helpful arms);
- every registered predictor of *which* effects survive failed.

This registration tests causes. Plan and rationale: `runs/reviews/round4/MECH_PLAN.md`. Tooling reports:
- `runs/reviews/round4/scratch/T19_kl_tooling_report.md` (M5, M6);
- `T19_grad_report.md` (M3, M4);
- `T19_repair_xlik_report.md` (M1, M2).

**Frozen scripts** (`prereg/scripts/p19/`; SHA-256):
- `score_prereg19_kl.py` dececfb4afb06e7ff9e8d29b4d78976f7661d20074c7a9e63cc7c5e458706fc6 (identical to `runs/theory/scratch/score_prereg19_kl.py`)
- `submit_prereg19_kl.sh` 0f58c0647483ae98b9df87630db0d29e9aebd4c38462f3fc6ae820bf627d89a1
- `submit_prereg19_drgrpo.sh` f63ed3a761c816a1c556a178a60db694403494bd273567e607ba2677b3950255
- `kl_travel.py` f37f7d7bd8d9dcd37a9cc8928f818decc074d8386ffd3d95a19d2e316e968000

The scorer imports the frozen prereg-15/17 libraries (`prereg/scripts/p17/`). Code: commit ca28859 adds the β > 0 resume fix (without it a preempted β > 0 run silently restarts from step 0) and `probe.py --merged_start`.

## Part I

### M5 — KL anchoring (GSM8K, Llama-3.2-1B-Instruct)
**Hypothesis.** RL erases SFT effects because nothing anchors it to its start.
- The one-index null, under which retention stays near 1, is exactly the optimum of KL-regularised RL (a tilt of each start; Prop. 1 of the paper).
- Our runs had β = 0. With a KL penalty to the start policy, retention should rise toward the null, and erasure should grow with the distance each start travels.

**Design.** As the registered Part X merged-fresh protocol (`prereg/scripts/p15/submit_prereg15x.sh`), with these changes:
- 300 steps;
- constant lr after 10 warm-up steps;
- checkpoints 100 / 200 / 300;
- β ∈ {0, 0.02, 0.1, 0.5}.

Everything else is unchanged:
- arms: gold-250 and teacher-250 (08 S′ SFT adapters, merged) and RL alone (fresh LoRA on the base);
- seeds 42–44;
- reward marked + stop.

Under merged-fresh, TRL's KL reference is each arm's own start (verified in smoke job 27671954: logged KL = 0 at steps 1–2, and reference logits equal the merged start).

**Evaluation.**
- probe.py, k = 4, T 0.6, top-p 0.95, 1,319 GSM8K test items, at the merged start (`--merged_start`) and at steps 100 / 200 / 300.
- The untrained start reuses prereg-14 Part A's untrained data (identical settings) pooled with E2, as in prereg 15.
- Per-item KL travelled (`kl_travel.py`): KL(π_t ‖ π_start) from the probe samples by teacher forcing, at T 0.6 (T 1 reported).

**Registered predictions** (lenient primary, marked secondary, per source; power from the synthetic dry run, job 27699394, 60 datasets per world):

| ID | Prediction | P(held \| anchoring raises b) | P(held \| b flat in β) |
|---|---|---|---|
| **M5a** (primary) | b(β = 0.5, 300) − b(β = 0, 300) ≥ 0.15, and its 95% CI lower bound > 0 | 1.00 / 1.00 | 0.00 / 0.03 |
| **M5c** (primary) | Erasure tracks distance travelled: the step-stratified Spearman of (1 − b) against the SFT start's mean per-item KL travelled, across β within each step, averaged over steps, has a 95% CI lower bound > 0 | 1.00 / 1.00 | 0.00 / 0.03 |
| M5b (secondary) | Anchoring restores the null: b(0.5, 300) lies inside the one-index tilt null's 95% band | 0.63 / 0.85 | 0.00 / 0.00 |
| M5a′ (secondary) | Spearman over the 4 β of b(300) ≥ 0.8 | 1.00 / 1.00 | 0.07 / 0.22 |

**Stated in advance.**
- M5a and M5c do **not** separate "anchoring changes where RL goes" (switch) from "KL only slows RL down" (brake). Both predict them.
- The matched-KL excess (M5d: erasure at β > 0 minus erasure at β = 0 at equal KL travelled) separates them in point estimate, but its power is ≤ 0.08 at 3 seeds. It is **reported, not tested**.
- **Guard.** If RL alone at β = 0.5 gains < 2 points (lenient, q_R − p_U at 300) over the untrained start, a held M5a is read as "RL stalled", not "anchoring". q_R − p_U is printed per β.

**Decision table.**

| Outcome | Reading |
|---|---|
| M5a and M5c hold (both sources) | Erasure is what unanchored RL does, and it grows with how far RL moves the policy. Retention depends on the KL budget, not on SFT content |
| M5a holds, M5c fails | KL preserves SFT effects, but not via distance travelled. Report |
| M5a fails | Anchoring does not preserve per-item SFT effects at these β: erasure is not explained by the absence of a KL term |
| M5b also holds | At β = 0.5 the dynamics reach the KL-optimal (tilt) regime |

### M6 — GRPO length bias in damaged starts (MATH L3–5, Gemma-3-1b-it)
**Hypothesis.** On starts with low reward (Gemma after gold or teacher SFT: accuracy 0.12–0.16), GRPO's per-sequence length normalisation favours longer wrong answers. Lengths then grow until they hit the 1,024-token cap, and repair stalls. Dr. GRPO removes that normalisation.

**Design.**
- The 17B Gemma gold and teacher merged-fresh runs, seeds 42–43, from the same 17B SFT adapters (`--reuse_from`; SFT not redone), with `--grpo_loss_type dr_grpo`.
- Everything else is as 17B: 300 steps, constant lr, marked + stop reward, 1,024 tokens.
- Evaluation as 17B: k = 8 at 150 / 300; the start and untrained evaluations reuse 17B's.

**Registered predictions** (lenient; no stop-rate exclusion, because truncation is the outcome):
- **M6a.** Each arm's training truncation share at logged steps 250–299 is ≤ the 17B GRPO RL-alone level + 0.05.
- **M6b.** Mean accuracy at step 300, dr_grpo minus 17B GRPO on the same SFT adapters (paired by seed), is > 0 with a 95% CI lower bound > 0, per arm.

**Reading.**
- M6a and M6b hold: the length bias of GRPO's loss is the cause of the stalled repair.
- M6a holds, M6b fails: length was a symptom; repair is limited otherwise. Part II's M1 (repair needs signal) then carries the explanation.
- M6a fails: the blow-up is not caused by length normalisation.

### Exclusions (M5)
The registered stop-rate rule (< 0.5 over the last 20 training steps) applies per run. An excluded SFT-arm run also leaves that seed's start cell for its β.

### Scoring
- `sbatch -p mit_normal -c 4 --mem=16G -t 04:00:00 --exclude=node2119,node3006 --wrap "… python -I runs/theory/scratch/score_prereg19_kl.py --unblind"`
- Output: `runs/theory/scratch/score_prereg19_kl_out.txt`. The SHA is checked against this file before scoring.

### What was seen before registration
- Smoke runs of ≤ 10 steps, whose accuracies were 8–80 samples each and are not data.
- 17B's Gemma results: the GRPO baseline that M6 is compared against, already scored (prereg 17).
- The tooling engineers ran one empty `python3` on the login node each; no code ran.

## Part IIa (registered 2026-11-04, before any M1 or M2 statistic was computed on real data)

Tooling report: `runs/reviews/round4/scratch/T19_repair_xlik_report.md`. Frozen scripts (`prereg/scripts/p19/`; SHA-256):
- `score_prereg19_repair.py` 6c6179de7075929f33ef92fa427d301c6228b5b0bb362768632359d45b978fd9
- `score_prereg19_xlik.py` 64860a2741b30f0ba06a60b5b87dd1cec7da1d0283c5ad797115ec2dff58c113
- `xlik.py` 2056a153ac6d081aa40ec74692ee922210964fb5ebaeadcf8de6ccf3ec24d019
- `p19_xlik_run.sh` 8964b13e67a6f1943557230c87790d6f296438ebfaff2e2b3e7d65653a78d115

Each is identical to its working copy in `runs/theory/scratch/`.

**Data status.** M1 and M2 use existing, arm-level-scored data:
- GSM8K Part A (prereg 15);
- MATH L3–5 17B (prereg 17), with Gemma including the runs excluded by 17B's stop-rate rule (post hoc arms).

Their per-item M1/M2 statistics have not been computed. Seen during tooling:
- schema only;
- one `results.csv` row with an already-reported arm gain;
- smoke-test job logs (timings and sanity counts), but not the smoke likelihoods.

### M1 — repair needs signal
**Hypothesis.** GRPO repairs an item damaged by SFT only where the damaged start still samples correct answers on it.

**Statistic.**
- C_ex: among hurt items, the contrast in damage-normalised repair between items whose SFT start had **no** correct sample on the selection half (Z) and items that had one (N).
- It is cross-fitted, and measured on the other half and the post-RL samples.
- "_ex" is the excess over the registered one-index tilt null, simulated through identical cells (definitions: report §2).

Calibration: 2,100 synthetic datasets at real sizes, job 27699259.

| ID | Set | Rule | P(held): signal-gated / uniform / tilt / memoryless own-shift |
|---|---|---|---|
| **M1-1** (primary) | GSM8K gold pool (gold-250 + gold-500, step 375); Gemma harmful pool (gold + teacher, all runs, post hoc arms; step 300) | power gate and C_ex > 0.10, each set separately | 1.00 / 0.00 / 0.00 / 0.60 (GSM8K), 1.00 (Gemma) |
| **M1-1b** (secondary) | same | M1-1 and C_exown > 0.05 (excess over a memoryless own-shift null): the gated repair is history-directed (undoing) | 1.00 / 0.00 / — / ≤ 0.05 |
| M1-1 (reported) | Llama MATH L3–5 gold | as M1-1 | 0.44: underpowered |
| **A** (secondary, weak) | the 6 damaged arms | A-1: Spearman(early zero-variance share in excess of RL alone, gap closed) ≤ −0.6; A-2: the two Gemma arms rank top 2 in that share | FP 0.047; power 0.36–0.86 (toy calibration). On GSM8K S′, TRL's logged `frac_reward_zero_std` substitutes for missing rollouts; the substitution is checked on the 17B runs |
| M1-3 | Gemma vs Llama | the composition share κ | descriptive only (0.00 power in every world) |

**Reading.**
- M1-1 holds in both sets: repair concentrates where the damaged start still samples correct answers, so Gemma's unrepaired damage reflects missing learning signal.
- M1-1b also holds: the signal-gated part is the undoing.
- M1-1 fails: repair is not signal-gated per item.

### M2 — behavioural convergence (cross-likelihood)
**Hypothesis.** Erasure means the two starts converge to the same per-item *solutions*, not only the same success rate.

**Design.**
- Teacher-force each role's reward-1 samples under the four models (untrained, SFT start, RL alone at the final step, SFT start at the final step) at T = 1.
- Statistics κ (convergence), u and c (likelihood-space routes); definitions in report §4.
- Under any reward tilt (the KL-regularised optimum), exactly κ = 1 and u = c = 0 at T = 1. Samples were drawn at T 0.6, so the null is approximate; its deviation was calibrated (job 27895185, 16,000 datasets).
- Arms: GSM8K gold-250 and teacher-250 (3 seeds); Llama MATH L3–5 gold and teacher (2 seeds) (primary). Gemma gold and teacher (2 seeds): post hoc, reported.

| ID | Rule (per arm) | Calibration |
|---|---|---|
| **X1** convergence beyond any tilt | HELD: κ ≤ 0.6 and CI upper bound < 0.9. REJECTED (tilt-like): CI lower bound > 0.6. Otherwise INCONCLUSIVE | false positive of HELD ≤ 0.01; power 0.93–1.00 at 50% convergence when the pre-RL models are separated, z(A_pre) ≳ 6 |
| **X2** route agreement | arms scored as undone (gold; Gemma teacher): u_share ≥ 0.65 and the CI of u > 0. Arms scored as matched (GSM8K and Llama teacher): 1 − u_share ≥ 0.65 and the CI of c > 0 | false positive ≤ 0.01; power 0.84–1.00 when separated |

**Reading.**
- If both GSM8K arms are INCONCLUSIVE (small z(A_pre)), M2 is uninformative at these sample sizes, and no "same solutions" claim is made either way.
- Shared styles between gold and teacher can make RL alone's move toward one start count as convergence for the other. Read u and c per arm against the other arm's, and report the gold↔teacher cross blocks.

### Scoring
- M2 runs: `bash runs/theory/scratch/p19_xlik_run.sh --submit` (mit_preemptable; one GPU job at a time).
- Then: `score_prereg19_xlik.py --unblind` and `score_prereg19_repair.py --unblind` (CPU via Slurm; SHAs checked first).
- Outputs: `runs/theory/scratch/score_prereg19_{xlik,repair}_out.txt`.

Part IIb (M3 expected and optimiser-preconditioned gradient; M4 gradient alignment vs dataset distance) will be appended before any of its statistics is computed.
