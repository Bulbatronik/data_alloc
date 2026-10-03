# 04 — Does GRPO's length normalisation hold back the longer own style? (registered 2026-10-02, before any of these runs)

**Theory.** `runs/theory/theory_v2.md`; Lemma L and Prop. A in the paper.
- Under TRL `loss_type="grpo"`, each rollout's weight is A/L. The own `## Step` style is 2.0–2.3× longer than the gold style. So its effective fitness gap is about half of Δp, by a measured factor r = Δp/Δ̃_grpo ≈ 1.8–2.0 per adapter.
- Under `dr_grpo` (constant normaliser), the full Δp acts. This is equivalent to multiplying the RL budget by r.
- H_null: Adam's normalisation removes the length weighting, so `dr_grpo` behaves like `grpo`.

**Runs.**
- Same five gold adapters as in `02_rl_budget_controls.md`.
- `--grpo_loss_type dr_grpo`, otherwise standard: G=8, 1 epoch, top-p 0.95, full test set.
- GRPO seeds gs1 and gs2.
- Cells:
  - a039, a068, a105, a184 at 4× (8e-5);
  - a068, a105, a184 at 2× (4e-5).
- 14 runs in total.
- "Recovered" means more than 50% of final greedy answers are `native`.

**Calls.** Use the only constant from the `grpo` ladder: the band (3.7%, 4.2%) × lr.
- Call "recover" if r·lr·w0 > 4.2.
- Call "no" if r·lr·w0 < 3.7.
- Otherwise the cell is unscored.

| Cell | r·lr·w0 | Call | Outcome under `grpo` |
|---|---|---|---|
| a039 4× | 2.9 | no | 0/2 |
| a068 4× | 4.9 | **yes** | 0/2 |
| a105 4× | 7.7 | yes | 2/2 |
| a184 4× | 14 | yes | 1/2 |
| a068 2× | 2.4 | no | 0/2 |
| a184 2× | 7.1 | **yes** | 0/2 |
| a105 2× | 3.8 | borderline (unscored) | 1/2 |

**Registered predictions.**
- **D1.** At least 9 of the 12 decisive runs go as called.
- **D2 (the decisive contrast).** In the two cells that failed under `grpo` (a068 4×, a184 2×), at least 3 of the 4 runs recover. H_null predicts at most 1 of 4.
- **D3.** For every adapter, the recovered fraction under `dr_grpo` is at least the fraction under `grpo` at the same lr.

**Caveats, stated in advance.**
- 46–89% of forced `## Step` answers continue with gold `<<>>` annotations, and the outcome classifier counts those as gold. Recovery therefore also needs selection at the continuation level, which the model does not include.
- Under a distributed encoding of the style, the a068 4× call becomes borderline (4.0). So a recovery at a068 4× is the sharper evidence for opening-encoded selection.

## Outcome (scored 2026-10-03 with `runs/theory/scratch/score_prereg04.py`)

All 14 runs completed; none diverged. Recovered / runs:

| Cell | Call | `dr_grpo` | `grpo` |
|---|---|---|---|
| a039 4× | no | 0/2 | 0/2 |
| a068 4× | yes | **0/2** | 0/2 |
| a105 4× | yes | 1/2 | 2/2 |
| a184 4× | yes | 2/2 | 1/2 |
| a068 2× | no | 0/2 | 0/2 |
| a184 2× | yes | **2/2** | 0/2 |
| a105 2× | unscored | 2/2 | 1/2 |

| Prediction | Result | Held? |
|---|---|---|
| D1 | 9/12 decisive runs as called (registered: at least 9) | Yes |
| D2 | 2 of 4 runs in the two contrast cells recovered: a184 2× 2/2, a068 4× 0/2 (registered: at least 3 of 4) | **No** |
| D3 | dr_grpo recovered less often than grpo in one cell, a105 4× (1/2 vs 2/2) | **No** |

**Reading:**
- Removing length normalisation raised recovery from 4/14 to 7/14 runs across these cells.
- a184, whose own-style continuations are the purest (46% gold hybrids), flipped from 1/4 to 4/4. That fits the length-weighted-fitness account.
- a068 never recovered in any of 9 runs (grpo 1×/2×/4×, dr_grpo 2×/4×), although its w0 is 0.68% and dr_grpo at 4× is the largest effective budget we tried. Its forced `## Step` answers are 78% gold hybrids.
- The length correction is real but does not explain every adapter. A continuation-level effect is unmodelled: own-style openings followed by gold-style content.
