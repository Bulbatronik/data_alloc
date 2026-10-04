# 10 — Geometry3K: per-item measurements before any training (registered 2026-10-04, before these runs)

**Context.**
- `runs/directions/complex_datasets.md`: dataset, model and pilot design.
- `runs/directions/sample_value.md`: the value model. SFT supplies missing knowledge units. RL amplifies units the student already covers, with signal ∝ p(1−p).
- Prediction to test later: on equally hard items, SFT value depends on *why* an item fails.

**Data.** Geometry3K (`hiyouga/geometry3k`), with gold diagram literals from Inter-GPS (MIT licence).
- Train pool: all 2101 items. Test: the 601 official test items.

**Models.**
- Student: Qwen2.5-VL-3B-Instruct.
- Teacher for SFT traces: Qwen2.5-VL-32B-Instruct.

**Prompt and grading.**
- Prompt style `boxed_user`: the instruction "Please reason step by step, and put your final answer within \boxed{}." goes in the user turn.
- It was chosen on 64 *training* items, untrained, greedy:
  - `boxed_user`: 97% of answers use `\boxed{}`, marked accuracy 32.8%.
  - system-message `boxed`: 70% and 17.2%.
- Grading is marked: `\boxed{}` answers, symbolic equivalence (`expressions_match`).

## Measurements

All sampling uses T=0.8, top-p 0.95, up to 1024 tokens (2048 for the teacher), with outputs in `runs/vlm/geo3k/m0/`.

| Name | Model | Items | Input | Samples per item |
|---|---|---|---|---|
| p_img | student | train pool | image | 8 |
| p_txt | student | train pool | question text only | 8 |
| p_cap | student | train pool | Inter-GPS diagram literals as text, no image | 8 |
| p_test | student | test | image | 8 |
| p_teacher | teacher | train pool | image | 4 |

The teacher's correct samples become the SFT traces.

**Item classes** (train pool):
- **perception-gap:** p_img ≤ 1/8 and p_cap ≥ 3/8. The student fails with the image and succeeds with the description.
- **knowledge-gap:** p_img ≤ 1/8, p_cap ≤ 1/8 and p_teacher ≥ 2/4. The student fails either way; the teacher solves it.
- **covered:** 2/8 ≤ p_img ≤ 6/8, the RL-informative zone.

## Predictions

- **M1.** Mean p_cap > mean p_img: the gold diagram description helps the 3B student.
- **M2.** Mean p_txt ≤ 0.10: most items need the diagram.
- **M3.** Mean p_teacher ≥ 0.45.

## Pilot gate

The training pilot (to be registered as `11`) uses gap-routed SFT arms. It is designed as in `complex_datasets.md` §4 only if the perception-gap class and the knowledge-gap class each have ≥ 300 train items. Otherwise the arms are redesigned before registration.

## Outcome
(appended after the runs)
