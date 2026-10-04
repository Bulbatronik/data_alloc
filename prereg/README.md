# Pre-registrations

Each registration is committed and pushed to GitHub **before** any of its runs are submitted. The push time on GitHub
is the registration time. Outcomes are appended later, in separate commits, under an "Outcome" heading; predictions
are never edited after registration.

- `01_before_2026-10-02.md` — registrations made before this policy (§1–§10). They were recorded locally before the
  runs but were **not** publicly timestamped, so readers should treat them as unverified.
- `02_rl_budget_controls.md` — top-p, matched-budget and RL-learning-rate controls (Llama-3.2-1B, gold traces).
- `03_r1_arm.md` — SFT on DeepSeek-R1-Distill-Qwen-1.5B traces, then GRPO (Llama-3.2-1B/3B, SmolLM2-1.7B), with
  survival predicted from ŵ and harm from the fitness-gap proxy Δp.
- `04_loss_normalisation.md` — the same ladder adapters under `dr_grpo` (no per-rollout length normalisation).
- `05_routing_and_sft_saturation.md` — routing in both directions (gold and teacher traces) and Ding et al.'s SFT-to-saturation rule.
- `06_stage_ladder.md` — base, math-mid-trained and instruct Llama-1B: does the training stage set the allocation curve's shape?
- `07_replay_floor_and_step_matched.md` — own-style replay floor (gate for building a method) and step-matched GRPO run to a plateau.
- `08_marked_reward.md` — RL with a marked-answer reward (no last-number fallback): core rerun and step-matched control.
- `09_method_gate.md` — gate for forced own-mode openings (competence, hybrids, REFT reach) and a second latent mode (Qwen2.5-Math code reasoning).

Definitions used throughout:
- w0 — exact P(answer starts with `## Step`) at T=0.8 (`probe.py --prefix`).
- ŵ — model-agnostic own-style share (`probe.py --own_style`; `runs/theory/style_diagnostic.md`).
- "recovered" — more than 50% of final greedy test answers are in the own style (`modes.classify`).
