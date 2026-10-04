# 06 — Stage ladder: base → mid-trained → instruct (registered 2026-10-03, before any GRPO on these models)

**Question.** Does the starting checkpoint's training stage set the shape of the SFT/RL allocation curve, as the style-survival account predicts?

**Models.**
- Llama-3.2-1B base (`unsloth/Llama-3.2-1B`).
- OctoThinker-1B-Short-Base, math mid-trained from Llama-3.2-1B (`OctoThinker/OctoThinker-1B-Short-Base`).
- Llama-3.2-1B-Instruct: existing curves, reference only.

The two base checkpoints have no chat template, so they use the plain `Question: … Answer:` prompt.

**Runs.** GSM8K pool of 1000.
- n ∈ {0 (GRPO alone), 100, 250, 1000 (SFT alone)}; random subsets.
- Gold traces and Qwen2.5-7B-Instruct traces.
- Seeds 42, 43, 44.
- Standard SFT (lr 1e-4) and GRPO (G=8, lr 2e-5, 1 epoch, `loss_type=grpo`), full test set.
- Outputs: `runs/gsm8k/<model>/seed_S` (gold, plus n=0) and `runs/gsm8k/<model>-qwen7b/seed_S`.

## Known before registering (seed-42 SFT-only scans, 300 test items, `runs/gsm8k/<model>-sftscan*/`)

Accuracy after SFT, by number of SFT examples:

| Model / trace source | 0 (untrained) | 50 | 100 | 250 | 500 | 1000 |
|---|---|---|---|---|---|---|
| Base, gold | 4.0 | 2.7 | 3.3 | 9.3 | 6.7 | 10.0 |
| Base, teacher | 4.0 | 4.0 | 6.0 | 9.7 | 9.7 | 15.0 |
| OctoThinker, gold | 21.3 | 15.0 | 22.7 | 41.0 | 39.3 | 42.0 |
| OctoThinker, teacher | 21.3 | 45.7 | 55.3 | 55.3 | 57.7 | 59.0 |

**OctoThinker's own style** (markdown `#### Step k:`; `modes.classify` labels it `native`):
- ŵ after gold SFT: 99.7% / 81% / 0.12% / 0.007% at n = 50 / 100 / 250 / 500. Library mass 0.37.
- Δp proxy (untrained accuracy minus full-SFT accuracy) is strongly negative: −21 points for gold and −38 for teacher. So the surviving foreign style is *better*.
- The untrained model stops within 512 tokens on only 16% of prompts.

**Base model.** No usable own style (library mass 0.07); it stops on 44% of prompts.

## Predictions

The style account says a cliff needs a better own style that SFT extinguishes. Neither base checkpoint has one, so their curves should not fall with n.

**Base Llama-3.2-1B**
- **B1 (SFT beats RL per item).** For both trace sources, mean final(n=1000) ≥ mean final(n=0).
- **B2 (no cliff).** For both trace sources, mean final(n=250) ≥ mean final(n=0) − 1.5.

**OctoThinker-1B**
- **O1 (no cliff; SFT helps).**
  - Gold: mean final(n=250) ≥ mean final(n=0) − 1.5.
  - Teacher: mean final(n=250) ≥ mean final(n=0) + 3.
- **O2 (a worse own style is selected against even while it survives).**
  - Gold n=100 (ŵ ≈ 81%): in all 3 seeds, the GRPO rollout share of `native` over the last 5 logged steps is below its share over the first 5 (`grpo/rollout_modes.csv`).
  - At n = 250 (both sources): final greedy `native` share < 5% in every run.
- **O3 (source order).** Teacher > gold at n = 100 and n = 250 (seed means).

**Instruct (existing, not a prediction).** A cliff with gold traces and a flat curve with teacher traces.

**Scoring.** The ladder supports "stage sets the curve shape" if B1, B2, O1 and O2 all hold. O3 is secondary.

## Outcome (scored 2026-10-04 with `runs/theory/scratch/score_prereg06.py`)

All 33 runs completed. The registered metric is standard (lenient) grading; marked grading is reported alongside (see `08`).

Mean final accuracy over 3 seeds, lenient / marked:

| Model | Source | n=0 | n=100 | n=250 | n=1000 (SFT only) |
|---|---|---|---|---|---|
| Base | gold | 4.8 / 1.9 | 8.6 / 7.6 | 9.2 / 8.4 | 10.7 / 10.4 |
| Base | teacher | | 10.0 / 8.1 | 10.1 / 9.8 | 11.9 / 11.5 |
| OctoThinker | gold | 53.8 / **21.6** | 58.4 / 58.2 | 56.4 / 56.3 | 43.7 / 43.7 |
| OctoThinker | teacher | | 56.6 / 56.3 | 59.1 / 58.9 | 57.6 / 57.1 |

(n=0 is GRPO alone, shared by both sources.)

| Prediction | Held? (lenient / marked) | Note |
|---|---|---|
| B1 | Yes / Yes | |
| B2 | Yes / Yes | |
| O1 | Yes / Yes | |
| O2 | **No** | At gold n=100 the `native` rollout share *rose* during GRPO (0.01–0.02 → 0.11–0.27); final `native` at n=250 reached 0.14 (registered: < 0.05) |
| O3 | **No** | Teacher > gold at n=250 (59.1 vs 56.4) but not at n=100 (56.6 vs 58.4) |

**Registered verdict:** O2 failed, so the ladder does **not** support "stage sets the curve shape" as registered.

**Notes:**
- O2's operationalisation was flawed. ŵ (opening library) put OctoThinker's own `#### Step` opening at 81% after gold n=100. But `modes.classify` labels such answers `gold` whenever they carry `<<>>` annotations, and labels `\boxed` answers `native`. The registered test therefore tracked a different mode from the one ŵ measured.
- Under marked grading, OctoThinker's GRPO alone collapses (21.6: unmarked answers), while SFT followed by GRPO reaches 56–59. For this mid-trained model, SFT supplies the answer format that RL alone does not find.
