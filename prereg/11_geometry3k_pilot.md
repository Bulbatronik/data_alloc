# 11 — Geometry3K pilot: does a sample's SFT value depend on *why* it fails? (registered 2026-10-05, before these runs)

**Theory.** `runs/directions/sample_value.md` (unit-bottleneck model):
- SFT supplies missing knowledge units; RL only amplifies covered units.
- Its hallmark prediction is an interaction between failure type and trace grounding.

**Measurements.** `10`; per-item table `runs/vlm/geo3k/m0/per_item.csv`.

**Redesign.** The `10` gate failed (fewer than 300 items per gap class). This design replaces `complex_datasets.md` §4.

**Common setup.**
- Student: Qwen2.5-VL-3B-Instruct, LoRA on the language model, `--prompt_style boxed_user`, `--reward marked`.
- Pool: the 2101 Geometry3K train items. Test: the 601 official test items, image input, greedy, 1024 tokens.
- SFT: lr 1e-4, **2 epochs**, max length 2048, gradient checkpointing.
- SFT traces are the Qwen2.5-VL-32B teacher's first correct sample, in two variants:
  - **as-is:** `runs/vlm/geo3k/pilot/teacher.jsonl`;
  - **grounded:** the same trace preceded by the gold Inter-GPS diagram description (`teacher_grounded.jsonl`), so the student learns to state what the diagram shows before reasoning.
- Inputs are built by `runs/vlm/geo3k/prep_pilot.py`. Id lists use `random.Random(0)` and are restricted to items with a correct teacher trace (1331 items).
- Seeds 42 and 43.

## Phase 1 — SFT only, class × grounding (12 runs; mechanism test without RL)

**Arms.** {class} × {as-is, grounded} × {seed 42, 43}, with `listed_sft --skip_grpo`. All three class lists have 131 items.

| Class | Items | Mean p_img |
|---|---|---|
| perception-gap | 131 (all eligible) | 0.05 |
| knowledge-gap | 131 | 0.04 |
| random | 131 | 0.31 |

"Gain" = test accuracy minus untrained test accuracy, averaged over seeds.

**Predictions.**
- **H1 (hallmark interaction).** Both must hold:
  - perception-gap: gain(grounded) − gain(as-is) ≥ 2.0;
  - knowledge-gap: |gain(grounded) − gain(as-is)| ≤ 1.0.

  A difficulty-only account predicts no class-specific grounding effect.
- **H2 (missing units carry SFT value).** knowledge-gap gain(as-is) ≥ random gain(as-is) + 1.0.

## Phase 2 — allocation at matched RL budget (8 runs)

**Arms.** All arms run exactly 150 GRPO steps (8 prompts × 8 samples, default linear schedule) on the items not used for SFT.

| Arm | SFT items (as-is traces) | Mean p_img of SFT items |
|---|---|---|
| R0 | none (`full_grpo`) | — |
| zero-pass→SFT | `zeropass300`: 300 items with p_img ≤ 1/8 | 0.05 |
| easy→SFT | `easy300`: 300 items with the highest p_img | 0.81 |
| random→SFT | `random300` | 0.34 |

Each × seeds 42, 43.

**Predictions.**
- **P2.** mean(zero-pass) − mean(easy) ≥ 3.0. On GSM8K the same contrast was ≈ 0 (prereg `05`).
- **P2b.** mean(zero-pass) ≥ mean(random) + 1.0.

**Not predicted, reported:** each SFT arm against R0.

## Outcome, Phase 1 primary (scored 2026-10-05 with `runs/theory/scratch/score_prereg11p1.py`; registered 1024-token evaluation)

All 12 runs completed. Gain = test accuracy minus the run's own untrained accuracy (seed 42 / 43):

| Class | As-is traces | Grounded traces | Grounded runs that stop within 1024 tokens |
|---|---|---|---|
| perception-gap | +8.0 / +7.5 (mean **+7.7**) | −2.8 / −3.0 (mean −2.9) | 0.68–0.69 |
| knowledge-gap | +3.8 / +7.2 (mean **+5.5**) | −12.3 / −13.8 (mean −13.1) | 0.41–0.44 |
| random | +5.3 / +6.7 (mean **+6.0**) | −10.8 / −4.7 (mean −7.7) | 0.56–0.63 |

As-is runs stop within 1024 tokens 0.93–0.95 of the time.

| Prediction | Result | Held? |
|---|---|---|
| H1 | perception grounded − as-is = −10.7; knowledge \|grounded − as-is\| = 18.6 | **No** |
| H2 | knowledge − random (as-is) = −0.5 | **No** |

**Reading:**
- The grounded arms are dominated by truncation; see the amendment below. The 2048-token secondary analysis is pending.
- Not registered: with as-is traces, perception-gap items give the largest SFT gain.

## Amendment (2026-10-05, after seeing Phase-1 seed 42 only; seed 43 and Phase 2 not yet seen)

Seed 42 showed the following:

| SFT traces | Writes the diagram description first | Stops within 1024 tokens |
|---|---|---|
| grounded | 100% | 44–69% |
| as-is | 0% | 93–94% |

So at the registered 1024-token evaluation, grounded arms mostly fail by truncation before `\boxed{}`, which confounds H1.

**Primary analysis:** H1 and H2 are scored exactly as registered, at 1024 tokens.

**Secondary analysis, post hoc and labelled as such:** every Phase-1 adapter is re-evaluated greedily on the full test set at 2048 tokens, and H1/H2 are reported again.

Also noted: untrained test accuracy differs between jobs (27.45 vs 28.12), from batched-decoding nondeterminism (about 0.7 points). Gains are therefore computed against each run's own untrained evaluation.

## Outcome, Phase 1 secondary — POST HOC (2048-token re-evaluation; scored 2026-10-05 with `runs/theory/scratch/score_prereg11.py`)

All 13 evaluations completed (untrained plus 12 adapters). Untrained accuracy: 27.6.

| SFT class | As-is gain | Grounded gain | Stop rate within 2048 tokens (as-is / grounded) |
|---|---|---|---|
| perception-gap | +8.3 | −1.8 | 0.99 / 0.76 |
| random | +6.7 | −7.4 | 0.98 / 0.65 |
| knowledge-gap | +4.5 | −12.7 | 0.98 / 0.59 |

- H1 is still not supported: grounded − as-is is −10.2 for perception-gap and −17.2 for knowledge-gap. H2 is −2.3.
- **Why grounding fails is not the token limit.** Grounded-SFT models always start by writing a diagram description, then hallucinate facts and loop: in unfinished answers single lines such as "EF = 8." repeat hundreds of times.
- So at 3B with 131 examples, teaching the model to *generate* the gold description breaks decoding. H1 cannot be tested with grounding as an output; it needs the description supplied as an input.
- Untrained evaluations vary between jobs (25.96–28.12) from batched-decoding nondeterminism, more than the 0.7 noted in the amendment.

**Reading:** both Phase-1 predictions of the unit-bottleneck model fail on Geometry3K. With as-is traces, perception-gap items give the largest SFT gain and knowledge-gap items the smallest.
