# 18 — Per-item catch-up: does the lagging start catch up, or does the route follow the SFT source?

**Registered 2026-10-25, before any prereg-18 statistic was computed on its test data, and before the lead opened the prereg-17 17B scoring output.**

**Frozen scripts** (`prereg/scripts/p18/`; SHA-256):
- `score_prereg18.py` fc1c659e2250ca935be984f3bbfdf6e85f7991e1dd8389c51c2764731e01bd18 (identical to `runs/theory/scratch/score_prereg18.py`)
- `run_p18.sh` 488fbb63fd2c7fde08cabe811c154c0641a06795bd0bca13b06dc4a6e68c2458

It imports the frozen prereg-15/17 libraries unchanged (`prereg/scripts/p17/`: `score_prereg17.py` 82743ff5…, `p15lib.py` 9c15684c…, `score_prereg15.py` 9def2afa…). Tooling and calibration report: `runs/reviews/round4/scratch/T18_tooling_report.md`.

## Hypothesis
- **What 15 and 17A found, at the arm level.** RL erases most of SFT's per-item effect: retention b ≈ 0.05–0.33 against one-index nulls ≈ 0.8–1.0. The lost retention 1 − b splits into undoing (the SFT start moves) and convergence (RL alone moves):
  - gold (harmful) arms are undoing-dominated;
  - every helpful arm (teacher, RFT, head start, T_unc) is convergence-dominated, under both graders.
- **H-catchup (post hoc reading of 15 Part A's sign split).** The route is set per item by the sign of the SFT effect: on items SFT hurt, the SFT start (behind) catches up; on items SFT helped, RL alone (behind) catches up. The arm-level route then follows from each arm's mix of helped and hurt items.
- **Competing account, H-arm.** The route is a property of the SFT source. Every item of a helpful arm converges, and every item of a gold arm is undone, whatever the item's sign.
- **The discriminating items** are each arm's sign-minority items, fixed *a priori by SFT source*:
  - the **helped** side of gold-type arms (gold, gold-linear);
  - the **hurt** side of every other arm (teacher, RFT, HS, T_unc).

  H-catchup predicts catch-up there; H-arm predicts the opposite (fall-back).

## What was seen before registration
- **15 Part A** (hypothesis-generating; reproduced exactly by this scorer, job 26836394):
  - agree sides: large catch-up excess, +0.61 lenient and +0.58 marked;
  - discriminating sides: near zero (frozen chord −0.02 lenient, +0.16 marked; location-matched −0.19 / +0.10).

  Part A sits near the decision boundary and is **not** a test here.
- **17A:** the prereg-17 arm-level outcomes (b, β, c, level per arm; commit 899d85c). They were used to shape the synthetic calibration worlds (Var_true, b, side sizes). No sign-split statistic of any 17A new arm has been computed.
- **17B:**
  - the prereg-17 17B scoring job ran on 2026-10-25, but the lead has not opened its output;
  - the tooling engineer saw one raw row of `gemma_mf_untrained.jsonl` (one untrained sample's `correct` field) during a schema check;
  - no statistic was computed.

## Design
**Test sets** (cells, comparators and registered stop-rate exclusions exactly as the frozen 17 scorer builds them; lenient primary, marked secondary):

| Set | Arms (vs comparator) | Step | Role |
|---|---|---|---|
| **17A_new** | RFT, HS, T_unc (vs S′ GRPO alone); gold-linear (vs RL alone, linear) | 375 | primary |
| **17B both_all** | gold, teacher, RFT in Llama-3.2-1B-I and Gemma-3-1B-it (each vs its family's GRPO alone), the two families pooled | 300 | primary |
| 17A_helpful; 17B llama_all, gemma_all, *_helpful; step 150 | as above | — | reported |

**Statistic** (`T18_tooling_report.md` §2):
- Per arm-side, the frozen cross-fitted Wald chord (`p15lib.wald_parts`, τ = 1/3) gives undoing U and convergence C parts of 1 − b_side.
- The catch-up contrast is d = (dS + dU)/|dpre|: d+ = C+ − U+ on the helped side, d− = U− − C− on the hurt side. d > 0 means the lagging start moved more.
- Arm-sides are pooled by summing parts.
- **Excess over the reference-group tilt null.** A common logit shift alone makes raw d positive on both sides, because helped and hurt items sit at mid-range success. Every test therefore uses d minus the median of d under a one-index tilt null. The null's shift is fitted on items with |p̂_S − p̂_U| < τ and simulated through the identical cells and estimators (200 simulations).
- **Gate per arm-side:** ≥ 40 items per arm on the side, 1 − b_side ≥ 0.15, |chord| ≥ 3 points. A side failing the gate is untestable. Pool membership is fixed on the observed data before any interval is read.
- **Pools:**
  - discriminating (DISC): the a priori minority sides above;
  - agree: the other sides, with ≥ 2 testable arm-sides.
- B = 2000 hierarchical bootstrap (seeds within lineage, then items).

## Registered predictions (per primary set; lenient primary, marked secondary)

| ID | Rule (point rules; the bootstrap CI is reported) | P(held \| H-catchup, 75% share) 17A / 17B; marked 17A / 17B | P(held \| H-arm) |
|---|---|---|---|
| **H18-1 DISC (exclusion test)** | DISC excess > −0.10 | 0.97 / 0.90; 0.82 / 1.00 | ≤ 0.03 |
| **H18-2 AD2 (catch-up)** | agree excess > 0.15 **and** DISC excess > −0.10 | 0.80 / 0.87; 0.82 / 0.97 | ≤ 0.03 |

- Calibration: 3,120 synthetic datasets (60 per world × size), jobs 26834798–805 and 26834879–885, aggregated by 26836399. Worlds:
  - catch-up 100% / 75%;
  - symmetric (each start moves halfway);
  - fall-back (the leader drops);
  - **H-arm** (AR: c = 0.8 helpful, 0.2 gold);
  - AR17 (prereg 17's dry-run worlds);
  - majority-only catch-up;
  - null;

  each with closing on the probability and the logit scale.
- **Known limits, stated in advance:**
  - H18-1 is an *exclusion* threshold: a pass means "not H-arm", not catch-up on the discriminating sides. A symmetric world passes 0.72–0.88.
  - H18-2 has false-positive rates of 0.10 / 0.05 (lenient 17A / 17B) and 0.10 / 0.13 (marked) in the logit-scale symmetric world. No threshold reaches ≥ 0.8 power and ≤ 0.05 there in every cell.
  - CI-based versions have power ≤ 0.60, so they are reported, not tested.
  - 17B's discriminating pool is mainly gold's helped side: the helpful arms' hurt sides are testable in about 55% of lenient datasets and almost never under marked grading.
  - 17A's RFT and T_unc are near-untestable under lenient grading.
- Thresholds rest on synthetic worlds whose 17B shapes are guesses.

## Decision table (lenient, both primary sets)

| Outcome | Reading | Paper |
|---|---|---|
| H18-1 fails in both sets | The route follows the SFT source (H-arm) | Routes are reported as source-level properties; no per-item catch-up claim |
| H18-1 holds in both, H18-2 fails in either | H-arm excluded; catch-up not established (symmetric or mixed) | "The route is not a fixed property of the source"; no catch-up claim |
| H18-2 holds in both | Per-item catch-up: the lagging start catches up, and arm-level routes follow the sign mix | Unifying claim, with the symmetric-world false-positive rate (≤ 0.13) stated |
| The sets disagree | Task- or protocol-dependent | Report both; no general claim |

## Reported, not predicted
- Per-arm and per-side U, C, d (raw, excess over both nulls, location-matched chord); side sizes; per family; step 150.
- Marked-grader versions of everything.
- Part A, re-run as the reference.

## Scoring (CPU via Slurm; `--unblind` refuses to run unless this file exists)
- `sbatch -p mit_normal -c 8 --mem=32G -t 04:00:00 --exclude=node2119,node3006 prereg/scripts/p18/run_p18.sh --part all --unblind --proto mf`
  - The wrapper runs the identical working copy `runs/theory/scratch/score_prereg18.py`; its SHA is checked before scoring.
- Output: `runs/theory/scratch/score_prereg18_out.txt`.

## Conduct notes (tooling engineer, disclosed)
- An empty `python3 -` heredoc was invoked once on the login node; no code ran.
- `du -sh` was run once on the engineer's own 38 MB mock directory.
- One raw untrained-Gemma 17B row was displayed during a schema check (see above).

## Outcome (scored 2026-10-25 with the frozen scorer; `runs/theory/scratch/score_prereg18_out.txt`)
**Run.** Job 26847185, `--part all --unblind --proto mf`; scorer fc1c659e. Cells and exclusions come from the frozen 17 code:
- 17A: RFT seed 43 excluded;
- 17B: Gemma gold seed 43 and teacher seeds 42 and 43 excluded, which leaves Gemma with RFT only (see the prereg-17 17B outcome, commit f3c898b).

**Verdicts (lenient primary; marked secondary).**

| Set | Grader | H18-1 DISC (excess over reference-group tilt null; held if > −0.10) | Agree pool excess | H18-2 AD2 |
|---|---|---|---|---|
| 17A_new (375) | lenient | **not held**: −0.119 [−0.480, +0.679]; sides HS−, T_unc−, gold-linear+ (RFT− untestable) | +0.625 [+0.336, +0.985] | **not held** |
| 17A_new (375) | marked | held: −0.096 [−0.469, +0.796]; sides RFT−, T_unc−, gold-linear+ | +0.584 [+0.447, +0.784] | held |
| 17B both_all (300) | lenient | **untestable** (no discriminating side passes the gate) | +0.882 [+0.613, +1.296] | **untestable** |
| 17B both_all (300) | marked | held: +1.031 [+0.599, +2.572]; the only side is Llama gold's helped side | +0.677 [+0.457, +0.919] | held |

**Why 17B lenient is untestable.**
- Llama gold's helped side has 24.5 items per arm (gate: 40).
- The helpful arms' hurt sides also fail the gate.
- Gemma contributes only RFT, whose per-item effects are too small to test.
- The calibration had flagged that 17B's discriminating evidence would come mainly from gold's helped side.

**Decision-table reading (lenient).**
- H18-1 fails in 17A, narrowly: −0.119 against −0.10, with a wide interval. 17B is untestable.
- No decision row applies exactly. The applicable conclusion is that **per-item catch-up is not supported under the registered grader**: on the discriminating items of 17A, the lagging start does not catch up beyond the null, as H-arm predicts. The paper makes no per-item catch-up claim and reports routes as arm-level properties.

**Marked grader.** Both tests pass. In 17B the pass rests on one side: Llama gold's helped items, which under marked grading are dominated by answer-format help that RL alone also learns. This supports a format-specific catch-up, not the per-item hypothesis.

**Reported.**
- **Agree sides:** every set shows large catch-up beyond the null (+0.58 to +0.88), as both hypotheses predict.
- **Part A** was not re-run in this scoring job (`--part all` excludes it). Its reference run before registration (job 26836394; `T18_tooling_report.md` §3) reproduced the published prereg-15 values exactly.

**Combined with prereg 17B's post hoc Gemma analysis.** Where teacher SFT is harmful (Gemma), its effect is undone like gold's. The route tracks whether the *arm* helped or hurt, not each item's sign: a post hoc observation, consistent with the H18-1 result in 17A.
