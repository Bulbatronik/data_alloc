#!/usr/bin/env python3
"""Score prereg/20: scale and generality of prereg-15 Part A (GSM8K; gold-250, teacher-250 (Qwen2.5-7B-I traces) and
RL alone; seeds 42-44; 375 GRPO steps; probe.py evaluations at T 0.6, k = 4 on all 1,319 test items).
  S1 (--design 3b):     Llama-3.2-3B-Instruct, LoRA r16, merged-fresh protocol (prereg 15 Part X), marked + stop reward,
                        constant lr after 10 warm-up steps; untrained start = one probe evaluation with k = 16.
  S2 (--design fullft): Llama-3.2-1B-Instruct, full fine-tuning (fp32 master weights) for SFT and GRPO; untrained start
                        = the pooled 1B untrained cell of prereg-15 Part A (probe 4 + E2 3 x 4 samples; same base model).

The registered prereg-15/17 libraries are imported UNCHANGED from prereg/scripts/p17/ (SHA-256 verified at start):
p15lib.py (cells, hierarchical bootstrap, EIV retention, decomposition, Wald sign split, memory projection, bins),
score_prereg15.py (one-index tilt / affine / symmetric-memory nulls, memory decision, base-success excess) and
score_prereg17.py (arm_stats: convergence share c = beta_U / (1 - b), level retention; ar_world for the dry run).

Usage (Slurm CPU only, never on the login node):
  python -I runs/theory/scratch/score_prereg20.py --dry_run synthetic [--power_reps 40 --procs 8]  # power at 3 seeds
  python -I runs/theory/scratch/score_prereg20.py --dry_run mock --design {3b,fullft}  # end-to-end on synthetic files
  python -I runs/theory/scratch/score_prereg20.py --design {3b,fullft} --schema_only   # artefacts, counts; no outcomes
  python -I runs/theory/scratch/score_prereg20.py --design {3b,fullft} --unblind       # only once prereg/20_*.md exists
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "prereg/scripts/p17"
sys.path.insert(0, str(FROZEN))
import p15lib as L  # noqa: E402
import score_prereg15 as S  # noqa: E402
import score_prereg17 as S17  # noqa: E402

FROZEN_SHA = {"p15lib.py": "9c15684cbdd0ec13e00f6cf36034b475ad51fb22a2b16dae97413735606edf0d",
              "score_prereg15.py": "9def2afa975ecfca4ad8769f23fdc6cf7c82ca275da4da5ab21be6b48ee32e31",
              "score_prereg17.py": "82743ff54c9c21afcfd23932aec609569e622b31b1e754136e4fb040b3e2052f"}
P, fmt, med = S.P, S.fmt, S.med
GR = L.GRADERS
SEEDS = (42, 43, 44)
STEPS = (75, 225, 375)
PRIMARY = 375
ARMS = ("gold_250", "teacher_250")

# ---------------------------------------------------------------- proposed registered constants (fixed before data)
REG20 = dict(
    NULL_GAP=0.25,        # P20-1: b < median(one-index tilt null b) - 0.25, with the 95% CI upper bound also below
    A1_GAP=0.10,          # prereg-15 A1's bar, reported for comparability
    C_GOLD_MAX=0.35,      # P20-2: c_gold <= 0.35 (undoing-dominated) ...
    C_TEACHER_MIN=0.65,   # ... c_teacher >= 0.65 (convergence-dominated) and c_teacher - c_gold > 0 with 95% CI > 0
    GATE_VD=0.004, GATE_1MB=0.15,  # a route is untestable for an arm if Var_true(D_pre) < 0.004 or 1 - b < 0.15
    GAIN_MIN_PTS=2.0,     # mean-gain retention (q_S - q_R gap / p_S - p_U gap, probability scale) if |pre gap| >= 2 pts
    STOP_MIN=0.5,         # exclusion: GRPO stop rate < 0.5 over the last 20 steps (rollouts.jsonl)
)
PART_A_1B = {  # prereg-15 Part A outcome (08 S', continuing adapters, n = 250 + 500 pooled, lenient, step 375): reference only
    "gold": dict(b=0.081, beta_S=-0.71, beta_U=0.21), "teacher": dict(b=0.334, beta_S=-0.10, beta_U=0.57)}

DESIGNS = {
    "3b": dict(model="Llama-3.2-3B-Instruct", probes="runs/probes/prereg20_3b",
               sft="runs/gsm8k/Llama-3.2-3B-Instruct-p20-sft{src}/seed_{s}/random_sft_250",
               grpo="runs/gsm8k/Llama-3.2-3B-Instruct-p20-mf{src}/seed_{s}/{meth}/grpo", untrained="probe"),
    "fullft": dict(model="Llama-3.2-1B-Instruct", probes="runs/probes/prereg20_fullft",
                   sft="runs/gsm8k/Llama-3.2-1B-Instruct-p20-ft-sft{src}/seed_{s}/random_sft_250",
                   grpo="runs/gsm8k/Llama-3.2-1B-Instruct-p20-ft{src}/seed_{s}/{meth}/grpo", untrained="pooled1b"),
}
SRC = {"gold_250": "", "teacher_250": "-qwen7b", "grpo": "-rl"}
PROBE_FILES = ("untrained", "sft", "s75", "s225", "s375")
_RX = re.compile(r"/gsm8k/Llama-3\.2-(?:1B|3B)-Instruct-p20-(?:sft|mf|ft-sft|ft)(-qwen7b|-rl)?/seed_(\d+)/"
                 r"(random_sft_250|full_grpo)/(sft|grpo)(?:/checkpoint-(\d+))?/?$")


def parse20(a):
    """probe adapter path -> (arm, seed, stage); stage 'pre' = after SFT (merged start for 3b), int = GRPO checkpoint."""
    if a is None:
        return ("U", 0, "pre")
    m = _RX.search(a)
    assert m, f"unparsed prereg-20 adapter {a}"
    src, seed, meth, stage, ck = m.groups()
    arm = {None: "gold_250", "-qwen7b": "teacher_250", "-rl": "grpo"}[src]
    assert (arm == "grpo") == (meth == "full_grpo"), a
    assert stage == "sft" or ck is not None, f"{a}: only grpo/checkpoint-N adapters are scored"
    return (arm, int(seed), "pre" if stage == "sft" else int(ck))


def grpo_dir(root, design, arm, s):
    return Path(root) / DESIGNS[design]["grpo"].format(src=SRC[arm], s=s, meth="full_grpo" if arm == "grpo" else "random_sft_250")


def verify_frozen(strict):
    ok = True
    for f, h in FROZEN_SHA.items():
        got = hashlib.sha256((FROZEN / f).read_bytes()).hexdigest()
        ok &= got == h
        P(f"# frozen {f}: {'OK' if got == h else 'MISMATCH ' + got}")
    if strict and not ok:
        sys.exit("frozen libraries differ from their registered hashes: refusing to score")


# ================================================================ loading
def load_design(root, design):
    """-> (evaluations {(arm, seed, stage): Eval}, untrained units, excluded {(arm, seed)}, stop rates)."""
    root = Path(root)
    pdir = root / DESIGNS[design]["probes"]
    ev = {}
    for f in PROBE_FILES:
        p = pdir / f"{f}.jsonl"
        if f == "untrained" and DESIGNS[design]["untrained"] != "probe":
            continue
        if not p.exists():
            P(f"  {p}: MISSING")
            continue
        for k, e in L.read_probe(str(p), "gsm8k", key_fn=parse20).items():
            assert k not in ev, f"{k} evaluated twice"
            ev[k] = e
    P(f"marked regrade vs stored 'correct': {L.AGREE[True]}/{sum(L.AGREE.values())} agree")
    assert L.AGREE[False] == 0, "grader port disagrees with the stored marked field (probe.py must run with --reward marked)"
    if DESIGNS[design]["untrained"] == "probe":
        assert ("U", 0, "pre") in ev, "untrained evaluation missing"
        units = [(0, [ev.pop(("U", 0, "pre"))])]
    else:  # prereg-15 Part A pooled 1B untrained cell: probe (4 samples) + E2's three training-job evaluations (12)
        unt = L.read_probe(str(root / "runs/probes/prereg14_a/untrained.jsonl"), "gsm8k")[None]
        e2 = root / "runs/gsm8k/Llama-3.2-1B-Instruct-censored-grpo"
        units = [(0, [unt])] + [(i + 1, [L.check_aligned(L.read_job(str(e2 / f"seed_{s}/base_predictions.jsonl"), "gsm8k"),
                                                          unt.ids, unt.gold, f"E2 base seed {s}")]) for i, s in enumerate(SEEDS)]
    bad, rates = set(), {}
    for arm in ARMS + ("grpo",):
        for s in SEEDS:
            f = grpo_dir(root, design, arm, s) / "rollouts.jsonl"
            if f.exists():
                rates[(arm, s)] = r = S.stop_rate_rollouts(f)[0]
                if r < REG20["STOP_MIN"]:
                    bad.add((arm, s))
    return ev, units, bad, rates


def build_cells(ev, units, bad):
    """U (lineage None, never resampled); '<arm>|pre' and '<arm>|<step>' share the lineage <arm> (an SFT start and its
    RL run are resampled together); 'grpo|<step>' lineage grpo. A cell needs >= 2 seeds after exclusions."""
    cells = {"U": L.Cell.from_evals("U", "gsm8k", None, units)}
    short = []
    for arm, stages in [(a, ("pre",) + STEPS) for a in ARMS] + [("grpo", STEPS)]:
        for st in stages:
            u = [(s, [ev[(arm, s, st)]]) for s in SEEDS if (arm, s, st) in ev and (arm, s) not in bad]
            if len(u) >= 2:
                cells[f"{arm}|{st}"] = L.Cell.from_evals(f"{arm}|{st}", "gsm8k", arm, u)
            else:
                short.append(f"{arm}|{st} ({len(u)} seeds)")
    if short:
        P("cells not scored (< 2 seeds; rerun with new seeds, as in 13):", short)
    return cells


# ================================================================ statistics
def quad(arm, t):
    return (f"{arm}|pre", "U", f"{arm}|{t}", f"grpo|{t}")


def arm_block(v, o, q, pre):
    """Every reported per-arm statistic: prereg-15's all_stats (b, beta_S, beta_U, b+/b-, memory M, bins; both scales),
    prereg-17's c (and logit c) and level retention, and the probability-scale mean-gain retention."""
    out = L.all_stats(v, o, [q], S.TAU, prefix=pre)
    out.update(S17.arm_stats(v, o, q, pre))
    g0, g1 = out[pre + "gap_pre_pts"], out[pre + "gap_post_pts"]
    out[pre + "gain_ret"] = g1 / g0 if np.isfinite(g0) and abs(g0) >= REG20["GAIN_MIN_PTS"] else np.nan
    return out


def stats20_fn(steps=STEPS, graders=GR, lean=False):
    """lean: the registered primary quantities only (b, beta's, c, memory parts) - used by the power dry run."""
    def fn(v):
        out = {}
        for o in graders:
            for t in steps:
                for arm in ARMS:
                    q = quad(arm, t)
                    if not all(c in v.d.cells for c in q):
                        continue
                    pre = f"{o}|{t}|{arm}|"
                    if lean:
                        E = L.eiv_from(L.eiv_parts(v, o, *q, scale="p"))
                        out.update({pre + k: E[k] for k in ("b", "beta_S", "beta_U", "VDpre")})
                        out[pre + "c"] = E["beta_U"] / (1 - E["b"]) if np.isfinite(E["b"]) and 1 - E["b"] >= REG20["GATE_1MB"] else np.nan
                        out[pre + "M"] = L.memory_from(L.memory_parts(v, o, *q))["M"]
                    else:
                        out.update(arm_block(v, o, q, pre))
                kt, kg = f"{o}|{t}|teacher_250|c", f"{o}|{t}|gold_250|c"
                if kt in out and kg in out:
                    out[f"{o}|{t}|c_teacher_minus_gold"] = out[kt] - out[kg]
        return out
    return fn


def run_nulls20(d, o, nsim, seed, kinds=("tilt", "affine", "memsym"), log=True):
    aq = {a: quad(a, PRIMARY) for a in ARMS if all(c in d.cells for c in quad(a, PRIMARY))}
    nulls, models = S.run_nulls(d, o, aq, {a: (a,) for a in aq}, nsim, S.TAU, seed=seed, kinds=kinds, log=log)
    return nulls


# ================================================================ decisions
def gate_ok(obs, pre):
    vd, b = obs.get(pre + "VDpre", np.nan), obs.get(pre + "b", np.nan)
    return np.isfinite(vd) and vd >= REG20["GATE_VD"] and np.isfinite(b) and 1 - b >= REG20["GATE_1MB"]


def rule_p1(b, hi, null_med, gap=REG20["NULL_GAP"]):
    return bool(np.isfinite(null_med) and b < null_med - gap and hi < null_med - gap)


def rule_p2(obs, ci_diff, o, t=PRIMARY):
    """-> (verdict, split part, order part)."""
    pg, pt = f"{o}|{t}|gold_250|", f"{o}|{t}|teacher_250|"
    if not (gate_ok(obs, pg) and gate_ok(obs, pt)):
        return "UNTESTABLE (Var_true(D_pre) < 0.004 or 1 - b < 0.15 for an arm)", False, False
    cg, ct = obs[pg + "c"], obs[pt + "c"]
    split = cg <= REG20["C_GOLD_MAX"] and ct >= REG20["C_TEACHER_MIN"]
    order = ct - cg > 0 and ci_diff[0] > 0
    return ("HELD" if split and order else "not held"), split, order


def memory_verdict(M, reps_M, null_M):
    """prereg-15 A4 rule: UNDOING if CI(M - null median) > 0 and M above the null's 97.5th percentile; CONVERGENCE if
    CI < 0 and M below its 2.5th percentile."""
    m0 = med(null_M)
    lo, hi = L.ci(np.asarray(reps_M) - m0)
    nlo, nhi = L.ci(null_M)
    if lo > 0 and M > nhi:
        return "UNDOING"
    if hi < 0 and M < nlo:
        return "CONVERGENCE"
    return "none"


def decide20(obs, reps, nulls, design):
    ci = lambda k: L.ci(reps[k]) if k in reps else (np.nan, np.nan)
    res = {}
    for o in GR:
        P(f"\n## prereg-20 registered predictions [{design}, {'lenient (PRIMARY)' if o == 'len' else 'marked (secondary)'}, step {PRIMARY}]")
        for arm in ARMS:
            pre = f"{o}|{PRIMARY}|{arm}|"
            nd = nulls.get(o, {}).get(("tilt", arm))
            nm = med(nd["b"]) if nd is not None else np.nan
            b, (lo, hi) = obs.get(pre + "b", np.nan), ci(pre + "b")
            held = rule_p1(b, hi, nm)
            P(f"  P20-1 contraction beyond the null, {arm}: b {fmt(b, (lo, hi))} vs one-index tilt null median {nm:+.3f} "
              f"(bar {nm - REG20['NULL_GAP']:+.3f}) -> {'HELD' if held else 'not held'}"
              f" | prereg-15 A1 bar (null - 0.10): {'met' if rule_p1(b, hi, nm, REG20['A1_GAP']) else 'not met'}"
              f" | logit b {fmt(obs.get(pre + 'L_b', np.nan), ci(pre + 'L_b'))} vs its null "
              f"{med(nd['L_b']) if nd is not None else np.nan:+.3f}")
            res[(o, "P20-1", arm)] = held
        dk = f"{o}|{PRIMARY}|c_teacher_minus_gold"
        v, split, order = rule_p2(obs, ci(dk), o)
        cg, ct = obs.get(f"{o}|{PRIMARY}|gold_250|c", np.nan), obs.get(f"{o}|{PRIMARY}|teacher_250|c", np.nan)
        fin = np.mean(np.isfinite(reps[dk])) if dk in reps else np.nan
        P(f"  P20-2 routes: c_gold {fmt(cg, ci(f'{o}|{PRIMARY}|gold_250|c'))} (<= {REG20['C_GOLD_MAX']}), c_teacher "
          f"{fmt(ct, ci(f'{o}|{PRIMARY}|teacher_250|c'))} (>= {REG20['C_TEACHER_MIN']}); c_teacher - c_gold {fmt(obs.get(dk, np.nan), ci(dk))} "
          f"(finite in {fin:.2f} of replicates) -> {v} (split {'ok' if split else 'fails'}; order {'ok' if order else 'fails'})")
        res[(o, "P20-2")] = v
        mv = {}
        for arm in ARMS:
            nd = nulls.get(o, {}).get(("memsym", arm))
            k = f"{o}|{PRIMARY}|{arm}|M"
            if nd is None or k not in obs:
                continue
            mv[arm] = memory_verdict(obs[k], reps[k], nd["M"])
            lo, hi = L.ci(np.asarray(reps[k]) - med(nd["M"]))
            P(f"  P20-3 history asymmetry, {arm}: M {obs[k]:+.3f}; symmetric-null median {med(nd['M']):+.3f}; M - null "
              f"{obs[k] - med(nd['M']):+.3f} [{lo:+.3f}, {hi:+.3f}] -> {mv[arm]}")
        held3 = mv.get("gold_250") == "UNDOING" and mv.get("teacher_250") == "CONVERGENCE"
        P(f"  P20-3 (gold UNDOING and teacher CONVERGENCE) -> {'HELD' if held3 else 'not held'}")
        res[(o, "P20-3")] = held3
        for arm in ARMS:
            pre = f"{o}|{PRIMARY}|{arm}|"
            P(f"  reported {arm}: beta_S {fmt(obs.get(pre + 'beta_S', np.nan), ci(pre + 'beta_S'))}; beta_U "
              f"{fmt(obs.get(pre + 'beta_U', np.nan), ci(pre + 'beta_U'))}; mean-gain retention (prob.) "
              f"{fmt(obs.get(pre + 'gain_ret', np.nan), ci(pre + 'gain_ret'))} (pre gap {obs.get(pre + 'gap_pre_pts', np.nan):+.2f} pts, "
              f"post gap {obs.get(pre + 'gap_post_pts', np.nan):+.2f}); level retention (logit) "
              f"{fmt(obs.get(pre + 'level_ret', np.nan), ci(pre + 'level_ret'))}; b+ {fmt(obs.get(pre + 'b_plus', np.nan), ci(pre + 'b_plus'))}"
              f"; b- {fmt(obs.get(pre + 'b_minus', np.nan), ci(pre + 'b_minus'))}; Var_true(D_pre) {obs.get(pre + 'VDpre', np.nan):.4f}")
            ndict = {k: v_ for (k, a), v_ in nulls.get(o, {}).items() if a == arm}
            val, cint, lab = S.bin_excess(obs, reps, ndict, pre)
            P(f"  reported {arm}: base-success excess over the tilt null (bins {lab}) {fmt(val, cint)}")
            src = arm.split("_")[0]
            P(f"  reference (1B Part A, continuing adapters, n 250+500 pooled): b {PART_A_1B[src]['b']:+.3f}, beta_S "
              f"{PART_A_1B[src]['beta_S']:+.2f}, beta_U {PART_A_1B[src]['beta_U']:+.2f}")
        P("  b by step: " + "; ".join(f"{arm} " + " ".join(f"{t}: {obs.get(f'{o}|{t}|{arm}|b', np.nan):+.3f}" for t in STEPS) for arm in ARMS))
    return res


def score(root, design, B, nsim):
    P(f"\n# prereg 20 — design {design} ({DESIGNS[design]['model']})")
    ev, units, bad, rates = load_design(root, design)
    P("GRPO stop rate, last 20 steps:", {f"{a}/{s}": round(r, 3) for (a, s), r in sorted(rates.items())},
      "| excluded:", sorted(bad) or "none")
    d = L.Design(build_cells(ev, units, bad))
    P("cells:", sorted(d.cells))
    t0 = time.time()
    obs, reps = L.bootstrap(d, stats20_fn(), B, seed=2001)
    P(f"bootstrap B = {B}: {time.time() - t0:.0f}s")
    nulls = {o: run_nulls20(d, o, nsim, seed=2002) for o in GR}
    for o in GR:
        for t in STEPS:
            for arm in ARMS:
                pre = f"{o}|{t}|{arm}|"
                if pre + "b" in obs:
                    P(f"\n  [{o}] step {t} {arm}")
                    S.print_group(obs, reps, pre, {k: v for (k, a), v in nulls[o].items() if a == arm} if t == PRIMARY else None)
                    for k in ("c", "L_c", "gain_ret", "level_ret"):
                        if pre + k in obs:
                            P(f"    {k:30s} {fmt(obs[pre + k], L.ci(reps[pre + k]))}")
    return decide20(obs, reps, nulls, design)


# ================================================================ schema only (never reads outcome fields)
def schema20(root, design):
    root = Path(root)
    D = DESIGNS[design]
    P(f"# schema-only, design {design}: artefacts and probe-file structure; no outcome field is read")
    for arm in ARMS + ("grpo",):
        for s in SEEDS:
            fs = []
            if arm != "grpo":
                sd = root / D["sft"].format(src=SRC[arm].replace("-rl", ""), s=s) / "sft"
                fs.append(("sft", (sd / "adapter_model.safetensors").exists() or (sd / "model.safetensors").exists()))
            g = grpo_dir(root, design, arm, s)
            fs += [(f"ckpt-{t}", (g / f"checkpoint-{t}").exists()) for t in STEPS]
            fs += [("rollouts", (g / "rollouts.jsonl").exists())]
            if design == "3b":  # merged-fresh runs only
                fs.append(("merged_start.json", (g / "merged_start.json").exists()))
            P(f"  {arm} seed {s}: " + ", ".join(f"{n} {'ok' if x else 'MISSING'}" for n, x in fs))
    for f in PROBE_FILES:
        p = root / D["probes"] / f"{f}.jsonl"
        if not p.exists():
            P(f"  {p.name}: MISSING")
            continue
        info, _ = L.schema_probe(str(p))
        keys = sorted({(parse20(a)[0], parse20(a)[2]) for a in info}, key=str)
        P(f"  {p.name}: {len(info)} adapters; rows {sorted({x['rows'] for x in info.values()})}; items "
          f"{sorted({len(x['items']) for x in info.values()})}; samples/item {sorted({max(x['samples']) + 1 for x in info.values()})}; "
          f"keys {sorted(set().union(*[x['keys'] for x in info.values()]))}; parsed {keys}")


# ================================================================ dry run: power at 3 seeds (synthetic AR worlds)
WORLDS = [
    ("1B Part A-like routes (gold c 0.2, teacher c 0.8 + unreachable level)",
     dict(gold=dict(c=0.2, sd=1.3, mu_r=-0.7), teacher=dict(c=0.8, sd=0.9, mu_r=0.0, mu_u=0.16, sd_u=0.6))),
    ("same routes, SFT effects 0.6x (3B-like ceiling)",
     dict(gold=dict(c=0.2, sd=0.8, mu_r=-0.7), teacher=dict(c=0.8, sd=0.55, mu_r=0.0, mu_u=0.1, sd_u=0.35))),
    ("same routes, half the RL rate (less contraction)",
     dict(gold=dict(c=0.2, sd=1.3, mu_r=-0.7), teacher=dict(c=0.8, sd=0.9, mu_r=0.0, mu_u=0.16, sd_u=0.6), kappa=0.0033)),
    ("no route difference (c 0.5 both)",
     dict(gold=dict(c=0.5, sd=1.3, mu_r=-0.7), teacher=dict(c=0.5, sd=0.9, mu_r=0.0, mu_u=0.16, sd_u=0.6))),
    ("reversed routes (gold c 0.8, teacher c 0.2)",
     dict(gold=dict(c=0.8, sd=1.3, mu_r=-0.7), teacher=dict(c=0.2, sd=0.9, mu_r=0.0, mu_u=0.16, sd_u=0.6))),
    ("one-index tilt (the null is true; prereg-15 synth_A 'tilt')", dict(tilt=True)),
]


def _power_worker(job):
    wi, rep, B, nsim = job
    name, spec = WORLDS[wi]
    rng = np.random.default_rng(20000 + 1000 * wi + rep)
    if spec.get("tilt"):
        d, _ = S.synth_A(rng, "tilt", arms=ARMS)
    else:
        arms = {"gold_250": dict(spec["gold"]), "teacher_250": dict(spec["teacher"])}
        d, _ = S17.ar_world(rng, 1319, arms, (PRIMARY,), {}, kappa=spec.get("kappa", 0.0066), k=4, nU=4)
    obs, reps = L.bootstrap(d, stats20_fn((PRIMARY,), ("len",), lean=True), B, seed=int(rng.integers(1 << 30)))
    nl = run_nulls20(d, "len", nsim, seed=int(rng.integers(1 << 30)), kinds=("tilt", "memsym"), log=False)
    ci = lambda k: L.ci(reps[k]) if k in reps else (np.nan, np.nan)
    r = {}
    for arm in ARMS:
        pre = f"len|{PRIMARY}|{arm}|"
        nm = med(nl[("tilt", arm)]["b"])
        b, (lo, hi) = obs[pre + "b"], ci(pre + "b")
        r[f"b {arm}"], r[f"null b {arm}"], r[f"c {arm}"], r[f"VDpre {arm}"] = b, nm, obs.get(pre + "c", np.nan), obs[pre + "VDpre"]
        r[f"CI width b {arm}"] = hi - lo
        r[f"rule P20-1 {arm}"] = float(rule_p1(b, hi, nm))
        r[f"rule A1-bar (null-0.10) {arm}"] = float(rule_p1(b, hi, nm, REG20["A1_GAP"]))
        r[f"memory {arm}"] = memory_verdict(obs[pre + "M"], reps[pre + "M"], nl[("memsym", arm)]["M"])
    r["rule P20-1 both"] = float(r["rule P20-1 gold_250"] and r["rule P20-1 teacher_250"])
    v, split, order = rule_p2(obs, ci(f"len|{PRIMARY}|c_teacher_minus_gold"), "len")
    r["rule P20-2 routes"], r["rule P20-2 split (point)"], r["rule P20-2 order (CI)"] = float(v == "HELD"), float(split), float(order)
    r["P20-2 untestable"] = float(v.startswith("UNTESTABLE"))
    r["rule P20-3 gold UNDOING"] = float(r["memory gold_250"] == "UNDOING")
    r["rule P20-3 teacher CONVERGENCE"] = float(r["memory teacher_250"] == "CONVERGENCE")
    r["rule P20-3 both"] = r["rule P20-3 gold UNDOING"] * r["rule P20-3 teacher CONVERGENCE"]
    return wi, {k: x for k, x in r.items() if not k.startswith("memory")}


def dry_synthetic(args):
    from multiprocessing import Pool
    P(f"# DRY RUN (power at 3 seeds): {args.power_reps} datasets per world; Part-A sizes for ONE arm per source: 1,319 items, "
      f"k = 4 per adapter, 3 SFT seeds, 3 RL-alone seeds, untrained 16 samples; hierarchical bootstrap B = {args.B_power}; "
      f"tilt and symmetric-memory nulls nsim = {args.nsim_power}; lenient only; step {PRIMARY}")
    jobs = [(wi, r, args.B_power, args.nsim_power) for wi in range(len(WORLDS)) for r in range(args.power_reps)]
    with Pool(args.procs) as pool:
        out = pool.map(_power_worker, jobs, chunksize=1)
    for wi, (name, _) in enumerate(WORLDS):
        rr = [x for w, x in out if w == wi]
        P(f"\n  world: {name} ({len(rr)} datasets)")
        for k in sorted(rr[0]):
            vals = np.array([x[k] for x in rr], float)
            if k.startswith(("rule", "P20-2 untestable")):
                P(f"    {k:40s} P = {np.nanmean(vals):.2f}")
            else:
                P(f"    {k:40s} mean {np.nanmean(vals):+.3f} SD {np.nanstd(vals):.3f} (finite {np.mean(np.isfinite(vals)):.2f})")


# ================================================================ dry run: end-to-end on synthetic files (mock root)
def mock_root(args):
    """Writes probe.py-format files (adapter paths as the submission scripts produce them) and rollouts.jsonl stop data
    from an AR world into a mock root, then scores it with the real loader and decision code."""
    design = args.design
    mr = Path(__file__).resolve().parent / "p20" / f"mock_{design}"
    shutil.rmtree(mr, ignore_errors=True)
    rng = np.random.default_rng(2020)
    N = 1319
    arms = {"gold_250": dict(c=0.2, sd=1.3, mu_r=-0.7), "teacher_250": dict(c=0.8, sd=0.9, mu_r=0.0, mu_u=0.16, sd_u=0.6)}
    d, T = S17.ar_world(rng, N, arms, STEPS, {}, k=4, nU=4)
    gold = [str(1000 + i) for i in range(N)]
    D = DESIGNS[design]

    def rows(adapter, p, k):
        for i in range(N):
            for s in range(k):
                ok = rng.random() < p[i]
                comp = f"Reasoning.\n#### {gold[i] if ok else -7}"
                yield {"model": "mock", "dataset": "gsm8k", "adapter": adapter, "example_id": i, "sample": s, "gold": gold[i],
                       "correct": bool(ok), "stopped": True, "completion": comp}

    def write(path, items):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")

    def adapter(arm, s, stage):
        if stage == "pre":
            return D["sft"].format(src=SRC[arm], s=s) + "/sft"
        return D["grpo"].format(src=SRC[arm], s=s, meth="full_grpo" if arm == "grpo" else "random_sft_250") + f"/checkpoint-{stage}"

    pdir = mr / D["probes"]
    pseed = lambda T_, s: np.clip(T_ + rng.normal(0, 0.02, N), 0, 1)  # small per-seed spread around the planted truth
    write(pdir / "sft.jsonl", (r for arm in ARMS for s in SEEDS for r in rows(adapter(arm, s, "pre"), pseed(T[f"{arm}|pre"], s), 4)))
    for t in STEPS:
        write(pdir / f"s{t}.jsonl", (r for arm in ARMS + ("grpo",) for s in SEEDS
                                     for r in rows(adapter(arm, s, t), pseed(T[f"{arm}|{t}"], s), 4)))
    if D["untrained"] == "probe":
        write(pdir / "untrained.jsonl", rows(None, T["U"], 16))
    else:
        write(mr / "runs/probes/prereg14_a/untrained.jsonl", rows(None, T["U"], 4))
        for s in SEEDS:  # training-job format: row i = item i // 4, sample i % 4
            write(mr / f"runs/gsm8k/Llama-3.2-1B-Instruct-censored-grpo/seed_{s}/base_predictions.jsonl",
                  ({k: v for k, v in r.items() if k not in ("adapter", "example_id")} for r in rows(None, T["U"], 4)))
    for arm in ARMS + ("grpo",):
        for s in SEEDS:
            stop = 0.3 if (arm, s) == ("teacher_250", 44) else 0.99  # exercises the exclusion path
            write(grpo_dir(mr, design, arm, s) / "rollouts.jsonl",
                  ({"step": st, "reward": 1.0, "stopped": bool(rng.random() < stop)} for st in range(1, 376) for _ in range(64)))
    P(f"# mock root written: {mr} (teacher_250 seed 44 has stop rate 0.3 -> excluded)")
    res = score(mr, design, args.B, args.nsim)
    P(f"\n# mock verdicts: {res}")
    if not args.keep_mock:
        shutil.rmtree(mr, ignore_errors=True)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=list(DESIGNS))
    ap.add_argument("--dry_run", choices=["synthetic", "mock"])
    ap.add_argument("--schema_only", action="store_true")
    ap.add_argument("--unblind", action="store_true", help="score the real runs; refused unless prereg/20_*.md exists")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--nsim", type=int, default=200)
    ap.add_argument("--B_power", type=int, default=200)
    ap.add_argument("--nsim_power", type=int, default=60)
    ap.add_argument("--power_reps", type=int, default=40)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--keep_mock", action="store_true")
    args = ap.parse_args()
    P(f"# score_prereg20.py args={vars(args)}\n# REG20={REG20}\n# tau={S.TAU:.4f}")
    verify_frozen(strict=args.unblind)
    t0 = time.time()
    if args.dry_run == "synthetic":
        dry_synthetic(args)
    elif args.dry_run == "mock":
        assert args.design, "--dry_run mock needs --design"
        mock_root(args)
    elif args.schema_only:
        assert args.design, "--schema_only needs --design"
        schema20(args.root, args.design)
    else:
        regs = sorted((Path(args.root) / "prereg").glob("20_*.md"))
        if not args.unblind or not regs:
            sys.exit("refused: pass --unblind, and only after prereg/20_*.md is registered (none found)" if not regs
                     else "refused: pass --unblind to compute outcome statistics")
        assert args.design, "--unblind needs --design"
        P(f"# registration: {regs[0].name}")
        P(f"\n# verdicts: {score(args.root, args.design, args.B, args.nsim)}")
    P(f"\n# done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
