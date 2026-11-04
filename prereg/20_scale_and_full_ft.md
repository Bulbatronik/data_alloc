# 20 — Does per-item erasure survive scale and full fine-tuning?

**Registered 2026-11-04, before any run was launched.** Tooling report: `runs/reviews/round4/scratch/T20_scale_report.md`.

**Frozen scripts** (`prereg/scripts/p20/`; SHA-256):
- `score_prereg20.py` 61298a8314b0b1d59bb5376d053649894c62fbd62cc5dc863932d7d0cccac745 (identical to `runs/theory/scratch/score_prereg20.py`)
- `submit_prereg20_3b.sh` 46e05ae908285778ab0924c9b2f97b2e0361a9310464adbe0f085c16d8271252
- `submit_prereg20_fullft.sh` 74977818de5205ef3efaa91f86396a1f0c2baa1ed55b1420cb936237593ff75b

The scorer imports the frozen prereg-15/17 libraries. Code: commit ca28859 adds `--fp32_master`, full-model reuse, checkpoint pruning, and `probe.py` full-model and merged-start loading.

**Motivation.** Every scored erasure result is for 1B models with LoRA. A main-track reader will ask whether it survives a larger model and full fine-tuning. While building S2 we found that the pipeline's earlier full fine-tuning path trained pure-bf16 weights, which round away most updates: 0.50% vs 87.6% of weights changed in 5 steps. The earlier full-fine-tuning runs (`runs/gsm8k/Llama-3.2-1B-Instruct-fullft*`, October 1) are therefore not valid evidence, and no current claim uses them.

## Designs (GSM8K; arms: gold-250, teacher-250 (Qwen2.5-7B-Instruct traces) and RL alone; seeds 42–44; the 1B seeds' splits)
- **S1. Llama-3.2-3B-Instruct, LoRA r16, merged-fresh.**
  - SFT is trained for 3B.
  - GRPO: 375 steps, constant lr after 10 warm-up steps, 8 prompts × 8 samples, T 0.8, β 0, reward marked + stop; checkpoints 75 / 225 / 375.
  - Evaluation: probe k = 4, T 0.6, top-p 0.95, 1,319 items, after SFT and at 75 / 225 / 375; untrained k = 16.
- **S2. Llama-3.2-1B-Instruct, full fine-tuning** (`--no_lora --fp32_master`).
  - SFT lr and GRPO lr as in the frozen script.
  - Otherwise as S1. The untrained start is prereg-15 Part A's pooled 1B cell.

## Registered predictions (per design; lenient primary, marked secondary; step 375)

| ID | Prediction | Power (dry run, job 27699550) |
|---|---|---|
| **P20-1** contraction | For gold-250 and teacher-250 separately: b < median(one-index tilt null b) − 0.25, with the 95% CI upper bound also below that value | 1.00 at Part A sizes; 0.00 under the tilt null; ≈ 0.47 if the SFT effects are 0.6× as large. An arm with Var_true(D_pre) < 0.02 is labelled LOW-POWER |
| **P20-2h** routes follow harm (primary route test) | Each arm whose mean SFT effect is negative (printed `gap_pre_pts` < 0, lenient) has matched share c ≤ 0.35. Each arm with a positive mean effect has c ≥ 0.65. If the two arms differ in sign, the c difference (helpful minus harmful; printed as c_teacher − c_gold, sign flipped if teacher is the harmful arm) has a 95% CI excluding 0 in the predicted direction. An arm is untestable if Var_true(D_pre) < 0.004 or 1 − b < 0.15 | as P20-2: 0.82–0.95 in route worlds, 0.00–0.03 in null worlds |
| P20-2 (secondary) routes by source | c_gold ≤ 0.35, c_teacher ≥ 0.65, c_teacher − c_gold > 0 with CI > 0 (the scorer's P20-2 line) | as above |
| **P20-3** undoing of harmful SFT | Gold's history-asymmetry test reads UNDOING (the scorer's per-arm P20-3 line for gold) | 0.80–1.00 |
| P20-3b (reported) | Teacher CONVERGENCE | 0.15–0.62: underpowered, reported only |

**Guard (RL activity).** If RL alone's lenient gain at step 375 (q_R − p_U, printed) is < 2 points or its CI includes 0, P20-1 for that design is "uninformative: RL inactive", not "not held". The GRPO learning rate of full fine-tuning follows a rule of thumb (2e-6).

**Generality reading.**
- "Erasure replicates at 3B" = P20-1 for both sources in S1.
- "Erasure survives full fine-tuning" = P20-1 for both sources in S2.
- "Routes follow harm at scale" = P20-2h and P20-3 in both designs.
- Anything else is design-dependent and reported as such.

**Reported, not predicted.** b, β_S, β_U, c, b± and M at 75 / 225 / 375; mean gaps in points; level retention (logit); logit b against its null; RL dose (q_R − p_U, q_S − p_S); stop and clip rates. Comparisons with the 1B LoRA Part A values are descriptive only (different protocol and pooled n).

**Exclusions.** The registered stop-rate rule (< 0.5 over the last 20 training steps), per run.

**Risks stated in advance.**
- At 3B the teacher arm may be near-neutral (earlier 3B teacher-250: −0.3 greedy points, one seed). P20-2h handles a sign change; a near-zero effect makes the arm untestable.
- Full-fine-tuning SFT may damage strongly, and large damage was not repaired within 300 steps on Gemma (prereg 17, post hoc).

## Scoring
- `sbatch -p mit_normal -c 4 --mem=16G -t 04:00:00 --exclude=node2119,node3006 --wrap "… python -I runs/theory/scratch/score_prereg20.py --design {3b,fullft} --unblind"`
- Output: `runs/theory/scratch/score_prereg20_{3b,fullft}_out.txt`. The SHA is checked first.
