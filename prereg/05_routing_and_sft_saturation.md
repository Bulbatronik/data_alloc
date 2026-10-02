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

## Outcome
(appended after the runs)
