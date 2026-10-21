#!/usr/bin/env python3
"""Mock repository root for an end-to-end test of score_prereg15.py --part {0,A,B,X,16} WITHOUT touching blinded data.
Every file the scorer reads is written with the real schema, real adapter paths and the real gold answers (from the
HF cache, in prepare_data's order), but with SYNTHETIC correctness drawn from a planted latent model (logit retention
0.5). Usage (Slurm CPU): python -I runs/theory/scratch/p15_mock.py <out_root>"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import route_sft_grpo as R  # noqa: E402  (load_problems: the pipeline's own item order and gold answers)

OUT = Path(sys.argv[1])
assert not OUT.exists(), f"{OUT} exists: the mock root must be written fresh (appends below)"
rng = np.random.default_rng(7)
SEEDS = (42, 43, 44)
G = "runs/gsm8k/Llama-3.2-1B-Instruct"
expit = lambda z: 1 / (1 + np.exp(-z))


def golds(dataset, n=None):
    ds = R.load_problems(dataset, "test").shuffle(seed=1234)
    ids = list(range(len(ds)))
    if n:
        ids = sorted(random.Random(0).sample(ids, n))  # probe.py --n_prompts
    return ids, [ds[i]["gold_answer"] for i in ids]


def completions(gold, p, k, unmarked=0.0):
    out = []
    for _ in range(k):
        if rng.random() < p:
            out.append(f"So we get {gold}." if rng.random() < unmarked else f"Reasoning.\n#### {gold}")
        else:
            out.append("Reasoning.\n#### 987654")
    return out


def probe_rows(dataset, adapter, ids, gold, p, k, unmarked=0.0):
    rows = []
    for e, g, pi in zip(ids, gold, p):
        for s, c in enumerate(completions(g, pi, k, unmarked)):
            rows.append({"model": "unsloth/Llama-3.2-1B-Instruct", "dataset": dataset, "adapter": adapter, "prompt_style": "train",
                         "on": "test", "temperature": 0.6, "force_prefix": None, "example_id": e, "sample": s, "gold": g,
                         "correct": R.answer_is_correct(c, g, True, dataset), "stopped": bool(rng.random() > 0.02), "completion": c})
    return rows


def job_rows(dataset, gold, p, k=4, unmarked=0.0):
    rows = []
    for g, pi in zip(gold, p):
        for s, c in enumerate(completions(g, pi, k, unmarked)):
            rows.append({"gold": g, "sample": s, "predicted": R.extract_answer(c, True, dataset),
                         "correct": R.answer_is_correct(c, g, True, dataset), "stopped": True, "completion": c})
    return rows


def write(path, rows):
    path = OUT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def rollouts(path, steps, base):
    rows = [{"step": t, "example_id": i, "mode": "prompt", "reward": float(rng.random() < base + 0.0005 * t), "correct": True,
             "stopped": True, "censored": False, "n_tokens": 100, "completion": ""} for t in range(steps) for i in range(64)]
    write(path, rows)


class World:
    """Planted latent model: p_U = expit(th); SFT arm p = expit(th + delta_arm); RL at step t:
    q_R = expit(th + c_t), q_S = expit(th + c_t + 0.5 delta_arm)."""

    def __init__(self, n):
        self.th = rng.normal(0, 2.0, n)
        self.delta = {}

    def d(self, arm):
        if arm not in self.delta:
            mu = 0.5 if arm.startswith("teacher") else -0.6
            self.delta[arm] = rng.normal(mu, 1.2, len(self.th))
        return self.delta[arm]

    def p(self, arm, step, seed):
        e = rng.normal(0, 0.15, len(self.th))
        if arm == "untrained":
            return expit(self.th)
        c = 0 if step in ("pre", 0) else 0.8 + 0.001 * int(step)
        if arm == "grpo":
            return expit(self.th + c + e)
        return expit(self.th + c + (1.0 if step in ("pre", 0) else 0.5) * self.d(arm) + e)


ARMS = [f"{s}_{n}" for s in ("gold", "teacher") for n in (50, 250, 500)]


def adapter(arm, seed, stage):
    if arm == "grpo":
        return f"{G}-marked-steps375/seed_{seed}/full_grpo/grpo/checkpoint-{stage}"
    src, n = arm.split("_")
    q = "-qwen7b" if src == "teacher" else ""
    return f"{G}{q}/seed_{seed}/random_sft_{n}/sft" if stage == "pre" else f"{G}-marked-steps375{q}/seed_{seed}/random_sft_{n}/grpo/checkpoint-{stage}"


# ---------------- Part A (GSM8K) + E2 untrained + S' trainer states + Part X
ids8, g8 = golds("gsm8k")
W8 = World(len(ids8))
write("runs/probes/prereg14_a/untrained.jsonl", probe_rows("gsm8k", None, ids8, g8, W8.p("untrained", "pre", 0), 4, unmarked=0.1))
for s in SEEDS:
    write(f"runs/gsm8k/Llama-3.2-1B-Instruct-censored-grpo/seed_{s}/base_predictions.jsonl", job_rows("gsm8k", g8, W8.p("untrained", "pre", s), unmarked=0.1))
    write(f"runs/gsm8k/Llama-3.2-1B-Instruct-censored-grpo/seed_{s}/full_grpo/predictions.jsonl", job_rows("gsm8k", g8, W8.p("grpo", 150, s)))
    rollouts(f"runs/gsm8k/Llama-3.2-1B-Instruct-censored-grpo/seed_{s}/full_grpo/grpo/rollouts.jsonl", 150, 0.5)
write("runs/probes/prereg14_a/sft.jsonl", [r for a in ARMS for s in SEEDS for r in probe_rows("gsm8k", adapter(a, s, "pre"), ids8, g8, W8.p(a, "pre", s), 4)])
for st in (75, 225, 375):
    write(f"runs/probes/prereg14_a/s{st}.jsonl", [r for a in ARMS + ["grpo"] for s in SEEDS for r in probe_rows("gsm8k", adapter(a, s, st), ids8, g8, W8.p(a, st, s), 4)])
for a in ARMS + ["grpo"]:
    for s in SEEDS:
        src = "-qwen7b" if a.startswith("teacher") else ""
        meth = "full_grpo" if a == "grpo" else f"random_sft_{a.split('_')[1]}"
        p = OUT / f"{G}-marked-steps375{src}/seed_{s}/{meth}/grpo/checkpoint-375/trainer_state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"log_history": [{"step": t, "completions/clipped_ratio": 0.6 if (a == "grpo" and s == 43) else 0.01, "reward": 0.5}
                                                 for t in range(10, 380, 10)]}))
for src, q in (("gold", ""), ("teacher", "-qwen7b")):
    for proto in ("cont", "mf"):
        for s in SEEDS:
            base = f"{G}-px-{proto}{q}/seed_{s}/random_sft_250"
            arm = f"{src}_250"
            write(f"{base}/predictions_after_sft.jsonl", job_rows("gsm8k", g8, W8.p(arm, "pre", s)))
            write(f"{base}/predictions.jsonl", job_rows("gsm8k", g8, W8.p(arm, 150, s)))
            rollouts(f"{base}/grpo/rollouts.jsonl", 150, 0.45)
for s in SEEDS:  # merged-fresh RL alone (full_grpo under px-mf only)
    mfr = f"{G}-px-mf/seed_{s}/full_grpo"
    write(f"{mfr}/predictions.jsonl", job_rows("gsm8k", g8, W8.p("grpo", 150, s)))
    rollouts(f"{mfr}/grpo/rollouts.jsonl", 150, 0.5)

# ---------------- Part 0 (MATH)
idsm, gm = golds("math")
WM = World(len(idsm))
M = "runs/math/Llama-3.2-1B-Instruct-marked-steps375"
rows = []
for s in SEEDS:
    write(f"{M}/seed_{s}/base_predictions.jsonl", job_rows("math", gm, WM.p("untrained", "pre", s)))
    write(f"{M}/seed_{s}/random_sft_250/predictions_after_sft.jsonl", job_rows("math", gm, WM.p("gold_250", "pre", s)))
    for arm, meth in (("grpo", "full_grpo"), ("gold_250", "random_sft_250")):
        write(f"{M}/seed_{s}/{meth}/predictions.jsonl", job_rows("math", gm, WM.p(arm, 375, s)))
        rollouts(f"{M}/seed_{s}/{meth}/grpo/rollouts.jsonl", 375, 0.4)
        for st in (75, 225, 375):
            rows += probe_rows("math", f"{M}/seed_{s}/{meth}/grpo/checkpoint-{st}", idsm, gm, WM.p(arm, st, s), 4)
write("runs/probes/prereg14_c/ckpt.jsonl", rows)

# ---------------- Part B (GSM-Symbolic, k = 1)
idsy, gy = golds("gsm_symbolic")
WY = World(len(idsy))
write("runs/probes/prereg14_b/untrained.jsonl", probe_rows("gsm_symbolic", None, idsy, gy, WY.p("untrained", "pre", 0), 1))
c1 = []
for a in ("gold_50", "gold_250", "teacher_250", "grpo"):
    for s in SEEDS:
        src = "-qwen7b" if a.startswith("teacher") else ""
        meth = "full_grpo" if a == "grpo" else f"random_sft_{a.split('_')[1]}"
        c1 += probe_rows("gsm_symbolic", f"{G}-marked{src}/seed_{s}/{meth}/grpo", idsy, gy, WY.p(a, 300, s), 1)
write("runs/probes/prereg14_b/c1ep.jsonl", c1)
for st in (225, 375):
    write(f"runs/probes/prereg14_b/s{st}.jsonl", [r for a in ("grpo", "gold_250", "teacher_250", "teacher_500") for s in SEEDS
                                                  for r in probe_rows("gsm_symbolic", adapter(a, s, st), idsy, gy, WY.p(a, st, s), 1)])

# ---------------- prereg 16 (file names of submit_prereg16.sh)
for ds, name, k, n in (("gsm_plus", "gsm_plus", 2, None), ("svamp", "svamp", 4, None), ("math", "math", 4, 600)):
    ids, gg = golds(ds, n)
    Wd = World(len(ids))
    write(f"runs/probes/prereg16/{name}_untrained.jsonl", probe_rows(ds, None, ids, gg, Wd.p("untrained", "pre", 0), k))
    for stage, tag in (("pre", "sft"), (375, "s375")):
        for part, arms in (("", [a for a in ARMS if a.startswith("gold")] + (["grpo"] if stage == 375 else [])),
                           ("_t", [a for a in ARMS if a.startswith("teacher")])):
            rows = [r for a in arms for s in SEEDS for r in probe_rows(ds, adapter(a, s, stage), ids, gg, Wd.p(a, stage, s), k)]
            if name == "math":
                write(f"runs/probes/prereg16/math_{tag}{part}.jsonl", rows)
            else:
                with open(OUT / f"runs/probes/prereg16/{name}_{tag}.jsonl", "a") as f:
                    f.writelines(json.dumps(r) + "\n" for r in rows)
print("mock root written to", OUT)
