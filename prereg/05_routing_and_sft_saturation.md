# 05 — Routing direction and "SFT until validation loss saturates" (registered 2026-10-02, before any of these runs)

**Model and settings.**
- Llama-3.2-1B-Instruct, GSM8K pool of 1000 problems.
- Standard SFT (lr 1e-4) and GRPO (G=8, lr 2e-5, 1 epoch, `loss_type=grpo`).
- Full test set (1319).

**Trace sources.**
- **gold:** GSM8K reference solutions.
- **teacher:** Qwen2.5-7B-Instruct traces (`runs/gsm8k/teachers/qwen7b.jsonl`).

**Reference values (existing runs, 3 seeds).** Final accuracy:

| | GRPO alone | n=250 | n=500 |
|---|---|---|---|
| gold, random | 52.7 | 46.4 | 44.0 |
| gold, adaptive (hard → SFT) | | ≈ random (±0.6) | ≈ random (±0.6) |
| teacher, random | | 55.2 | 54.5 |

## E2 — routing in both directions (18 new runs)

**Routing rules.**
- `adaptive` (hard → SFT): the existing router sends never-solved and always-solved probe problems to SFT and keeps the GRPO-informative ones for RL. This is the direction of He et al. (2606.04466) and ReLIFT.
- `easy` (easy → SFT): new. The highest probe pass rate goes to SFT, ties broken by the most familiar gold trace. This is the direction of DeReason (2603.11193).

**Runs.**
- gold: `easy_sft_{250,500}`, seeds 42–44 (6 runs).
- teacher: `adaptive_sft_{250,500}` and `easy_sft_{250,500}`, seeds 42–44 (12 runs).
- Output: `runs/gsm8k/Llama-3.2-1B-Instruct{,-qwen7b}-routing/seed_S/`.

**Predictions.**
- **E2a (null with gold).** |mean final(easy) − mean final(random)| < 1.5 points at both n.
- **E2b (null with teacher traces).** |mean final(rule) − mean final(random)| < 1.5 points for both rules at both n.
- The literature's alternative predicts at least 2 points for its preferred direction. Any cell with |Δ| ≥ 1.5 counts against our null.

## E3 — Ding et al.'s rule (2512.11470)

**E3a: validation loss vs final accuracy (existing adapters, about 1 GPU-h).**
- Gold adapters: the main curve (n = 50, 100, 250, 500, 750, full), seeds 42–44. NLL is measured on 256 GSM8K-test gold solutions.
- Teacher adapters: the `-qwen7b` curve (n = 50, 100, 250, 500), seeds 42–44. NLL is measured on teacher traces of 256 problems from each adapter's own GRPO bucket (held out from its SFT).
- Statistic: Spearman ρ between per-adapter validation NLL and that run's final accuracy. Ding et al. report r ≈ −0.92: lower loss, higher final accuracy.
- **E3a-1 (rule fails for gold).** ρ > 0: more gold SFT lowers validation loss while final accuracy falls.
- **E3a-2 (teacher).** ρ ≤ 0.

**E3b: Ding et al.'s design (12 runs).**
- `full_sft` with `--sft_epochs` ∈ {1, 2, 4}, then GRPO on the same 1000 prompts (`--grpo_on_sft_prompts`).
- gold vs teacher, seeds 42 and 43.
- **E3b-1.** Gold: the mean final at each epoch count is at most 50.0, below GRPO alone (52.7). The own style is extinct (w0 ≈ 3e-6) and does not recover.
- **E3b-2.** Gold: mean final(1 epoch) ≥ mean final(4 epochs).
- **E3b-3.** Teacher: the mean final at each epoch count is at least 52.7.

## Outcome (scored 2026-10-03 with `runs/theory/scratch/score_prereg05.py`)

All 30 runs completed; none diverged.

**E2 — routing direction.** Mean final over 3 seeds; difference from random routing at the same n:

| Source | Rule | n=250 | n=500 |
|---|---|---|---|
| gold | adaptive (hard → SFT) | +0.7 | −0.4 |
| gold | easy (easy → SFT) | −0.6 | **−2.7** |
| teacher | adaptive | −1.2 | +0.1 |
| teacher | easy | +0.3 | −1.2 |

- **E2a (gold null): no.** Easy-to-SFT at n=500 lost 2.7 points (41.3 vs 44.0; SDs 1.6 and 2.1). The other gold cells hold.
- **E2b (teacher null): yes.** All four cells are within 1.5 points.

**E3a — validation NLL vs final accuracy (Spearman ρ).**
- **E3a-1: yes.** Gold, 18 adapters: ρ = +0.96. More gold SFT lowers held-out gold NLL (0.660 → 0.573) while final accuracy falls (56.6 → 38.7). Ding et al.'s rule picks the worst dose.
- **E3a-2: yes.** Teacher, 12 adapters: ρ = −0.16.

**E3b — full SFT for k epochs, then GRPO on the same 1000 problems** (2 seeds; mean final, with after-SFT in parentheses):

| Source | 1 epoch | 2 epochs | 4 epochs |
|---|---|---|---|
| gold | 47.6 (39.7) | 47.4 (40.2) | 40.4 (36.5) |
| teacher | 56.1 (54.0) | 56.9 (54.4) | 54.4 (53.0) |

- **E3b-1: yes.** Gold stays at or below 50.0.
- **E3b-2: yes.** Gold at 1 epoch (47.6) is at least gold at 4 epochs (40.4).
- **E3b-3: yes.** Teacher stays at or above 52.7.

**Not registered:** with teacher traces, SFT on all 1000 problems followed by GRPO on the same problems (56.1–56.9) beats every disjoint split (53.4–55.2) and GRPO alone (52.7).
