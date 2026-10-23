#!/usr/bin/env python3
"""Score prereg/17 (17A: GSM8K new arms under the 08 S' protocol; 17B: MATH L3-5 generality, two student families).
A separate file because the prereg-15 scorer (score_prereg15.py, p15lib.py) is registered and frozen (byte-verified
copy in prereg/scripts/p15/); this file imports both unchanged and reuses their estimators, nulls and bootstrap.

Usage (Slurm CPU only):
  python runs/theory/scratch/score_prereg17.py --dry_run synthetic        # AR worlds: recovery + power of every rule
  python runs/theory/scratch/score_prereg17.py --part {17A,17B} --schema_only
  python runs/theory/scratch/score_prereg17.py --part {17A,17B} --unblind  # after registration and the runs

Conventions are prereg 15's: lenient primary, marked secondary; probability scale primary, smoothed logit reported;
halves = samples < k//2; hierarchical bootstrap (seeds within lineage, then items), B = 2000; one-index (tilt) null
fitted on the untrained start's own pair, 200 replicates. c = beta_U / (1 - b) is the convergence share and
u = 1 - c the undoing share (identities; W_attractor_note.md section 3); level retention = post / pre mean logit gap.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p15lib as L  # noqa: E402
import score_prereg15 as S  # noqa: E402

P, fmt, med = S.P, S.fmt, S.med
SEEDS = (42, 43, 44)
GR = L.GRADERS

# ---------------------------------------------------------------- registered constants (fixed before any 17 data)
REG17 = dict(
    # 17A P17-1 clock (calibrated in the synthetic dry run, section 2)
    CLOCK_EPS=0.10,        # |b_goldlin(375) - b_gold250 interpolated at the same cumulative lr| <= 0.10
    STEP_GAP=0.10,         # and b_goldlin(375) - b_gold250(step 375) >= 0.10 (the step clock is rejected)
    PROBE_EPS=0.12,        # early-probe forecast |b(375) - b(75)^(tau375/tau75)| <= 0.12 (goldlin)
    # P17-2 routes (W's thresholds) and the falsification rule
    # RFT recalibrated from W's (0.8, -0.10): with a true convergence-only RFT those held in 0.48 of datasets (beta_S SD 0.13)
    RFT_C=0.6, RFT_BS=-0.25, HS_C=0.9, HS_BS=-0.05, UNDO_BS=-0.30, UNDO_CI=-0.10,
    # P17-3 semigroup (HS)
    SEMI=(0.7, 1.3), SEMI_FAIL=(0.5, 1.5),
    # P17-5 ordering
    RFTHS_B375=0.25, COV_RHO=0.5,   # 0.25 (W: 0.15): b(375) SD 0.08 (RFT) and 0.15 (HS) in the dry run
    # P17-6 bounds (corrected for the smoothed-logit estimator), P17-7 level
    AMP=0.10, LEVEL_MAX=0.3, LEVEL_MIN_PRE=0.10,
    # power gates (an arm-statistic is 'untestable' below these)
    GATE_VD=0.004, GATE_1MB=0.15,
    # 17B
    B_NULL_GAP=0.10, C_GOLD_MAX=0.35, C_CONV_MIN=0.65, C_TOL=0.15, LEVEL_DIFF=0.0,  # margins: 0.5 split gave 0.63 false positives
)

# ================================================================ clocks (W_attractor_retro.py: HF schedules, warm-up 10)
def tau_const(t, warm=10):
    return sum(min(1.0, k / warm) for k in range(t))


def tau_lin(t, total, warm=10):
    return sum((k / warm) if k < warm else max(0.0, (total - k) / (total - warm)) for k in range(t))


STEPS17A = (75, 150, 225, 300, 375)
TAU_C = {t: tau_const(t) for t in STEPS17A}
TAU_L = {t: tau_lin(t, 375) for t in STEPS17A}

# ================================================================ statistics shared by 17A and 17B
def arm_stats(v, o, quad, prefix):
    """b, beta_S, beta_U, c, logit b, level retention, Var_true(D_pre), gaps for one (S, U, PS, PR) quadruple."""
    r = {}
    Ep = L.eiv_from(L.eiv_parts(v, o, *quad, scale="p"))
    El = L.eiv_from(L.eiv_parts(v, o, *quad, scale="l"))
    for k in ("b", "beta_S", "beta_U", "VDpre", "gap_pre_pts", "gap_post_pts", "rl_alone_pts", "rl_sft_pts"):
        r[prefix + k] = Ep[k]
    r[prefix + "L_b"], r[prefix + "L_beta_S"], r[prefix + "L_beta_U"] = El["b"], El["beta_S"], El["beta_U"]
    r[prefix + "L_c"] = El["beta_U"] / (1 - El["b"]) if np.isfinite(El["b"]) and 1 - El["b"] >= REG17["GATE_1MB"] else np.nan
    r[prefix + "c"] = Ep["beta_U"] / (1 - Ep["b"]) if np.isfinite(Ep["b"]) and 1 - Ep["b"] >= REG17["GATE_1MB"] else np.nan
    lpre, lpost = El["gap_pre_pts"] / 100, El["gap_post_pts"] / 100
    r[prefix + "L_gap_pre"], r[prefix + "L_gap_post"] = lpre, lpost
    r[prefix + "level_ret"] = lpost / lpre if np.isfinite(lpre) and abs(lpre) >= REG17["LEVEL_MIN_PRE"] else np.nan
    return r


def semigroup_parts(v, o, hs_post, R_t, R_t75):
    """P17-3: EIV slope of y = q_HS(t) - q_R(t) on x = q_R(t+75) - q_R(t), cross-fitted so that y and x never share
    q_R(t) samples (y uses one half, x the other); denominator Var_true(x) = Cov(x^h0, x^h1)."""
    qh = v.m(hs_post, o)[2]
    r0, r1, _ = v.m(R_t, o)
    s0, s1, s = v.m(R_t75, o)
    n = float(len(qh))
    num = 0.5 * n * (L.cov(qh - r0, s - r1) + L.cov(qh - r1, s - r0))
    den = n * L.cov(s0 - r0, s1 - r1)
    return num, den


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(L.rank(x[ok]), L.rank(y[ok]))[0, 1])


def unsolved_share(v, o, arm_pre, U="U"):
    """P17-5d (W's coverage share): share of Var_true(D_pre) on items the untrained start never solves in its
    selection samples (untrained units 0 and 1), with D_pre measured on disjoint untrained units (2, 3) and the arm's
    halves. Requires the 17A untrained cell (probe 4 + E2 3 x 4 samples)."""
    cell = v.d.cells[U]
    C = cell.C[o] if v.J is None else cell.C[o][:, :, v.J[cell.set]]

    def props(units):
        num = sum(C[h, u] for u in units for h in (0, 1))
        den = sum(cell.Nh[h, u] for u in units for h in (0, 1))
        return num / den
    sel = props([0, 1])
    s0, s1, _ = v.m(arm_pre, o)
    D1, D2 = s0 - props([2]), s1 - props([3])
    prod = (D1 - D1.mean()) * (D2 - D2.mean())
    tot = prod.sum()
    return float(prod[sel == 0].sum() / tot) if abs(tot) > 1e-12 else np.nan


# ================================================================ 17A
ARMS17A = ("rft", "hs", "tunc", "goldlin")
REF17A = ("gold_250", "teacher_250")
_RX17A = re.compile(r"/gsm8k/Llama-3.2-1B-Instruct-p17a-(rft|hs|tunc|goldlin|grpolin)/seed_(\d+)/"
                    r"(random_sft_250|listed_sft|full_grpo)/(sft|grpo)(?:/checkpoint-(\d+))?/?$")
P17A = "runs/probes/prereg17a"


def parse17a(a):
    m = _RX17A.search(a)
    if m is None:
        f = S.parse_gsm8k_adapter(a)  # 08 S' reference checkpoints (150 / 300)
        return (f[1], f[2], f[3])
    arm, seed, _, stage, ck = m.groups()
    return (arm, int(seed), "pre" if stage == "sft" else int(ck))


def quads17a(cells, step):
    """{arm: quad}: new arms vs S' GRPO alone (constant lr); goldlin vs the linear-schedule RL alone; S' reference."""
    q = {}
    for arm in ("rft", "hs", "tunc") + REF17A:
        qq = (f"{arm}|pre", "U", f"{arm}|{step}", f"grpo|{step}")
        if all(c in cells for c in qq):
            q[arm] = qq
    qq = ("gold_250|pre", "U", f"goldlin|{step}", f"grpolin|{step}")
    if all(c in cells for c in qq):
        q["goldlin"] = qq
    return q


def stats17a_fn(steps=STEPS17A, graders=GR):
    def fn(v):
        out = {}
        for o in graders:
            for t in steps:
                for arm, q in quads17a(v.d.cells, t).items():
                    out.update(arm_stats(v, o, q, f"{o}|{t}|{arm}|"))
            # P17-1 clock: gold_250 (constant lr) interpolated at goldlin's cumulative lr
            for t in steps:
                tl = TAU_L[t]
                lo = max([s for s in steps if TAU_C[s] <= tl], default=None)
                hi = min([s for s in steps if TAU_C[s] >= tl], default=None)
                kl, kh = f"{o}|{lo}|gold_250|b", f"{o}|{hi}|gold_250|b"
                if lo is not None and hi is not None and kl in out and kh in out:
                    w = 0.0 if hi == lo else (tl - TAU_C[lo]) / (TAU_C[hi] - TAU_C[lo])
                    out[f"{o}|{t}|clock|b_ref_tau"] = (1 - w) * out[kl] + w * out[kh]
                    if f"{o}|{t}|goldlin|b" in out:
                        out[f"{o}|{t}|clock|delta_tau"] = out[f"{o}|{t}|goldlin|b"] - out[f"{o}|{t}|clock|b_ref_tau"]
                if f"{o}|{t}|goldlin|b" in out and f"{o}|{t}|gold_250|b" in out:
                    out[f"{o}|{t}|clock|delta_step"] = out[f"{o}|{t}|goldlin|b"] - out[f"{o}|{t}|gold_250|b"]
            if f"{o}|75|goldlin|b" in out and f"{o}|375|goldlin|b" in out:
                b75 = max(out[f"{o}|75|goldlin|b"], 1e-6)
                out[f"{o}|clock|probe_forecast"] = b75 ** (TAU_L[375] / TAU_L[75])
                out[f"{o}|clock|probe_err"] = out[f"{o}|375|goldlin|b"] - out[f"{o}|clock|probe_forecast"]
            # P17-3 semigroup (HS): registered on t = 75 and 150 pooled (moments summed); later steps are reported only,
            # because RL alone's increments q_R(t + 75) - q_R(t) are nearly flat by t >= 150 (synthetic dry run: the
            # EIV denominator vanishes and the t = 150 / 300 slopes have SD 1-10)
            pn = pd_ = 0.0
            for t in (75, 150, 225, 300):
                c = (f"hs|{t}", f"grpo|{t}", f"grpo|{t + 75}")
                if all(x in v.d.cells for x in c):
                    num, den = semigroup_parts(v, o, *c)
                    out[f"{o}|{t}|hs|semigroup"] = num / den if den > 0 else np.nan
                    if t in (75, 150):
                        pn, pd_ = pn + num, pd_ + den
            if pd_ > 0:
                out[f"{o}|hs|semigroup_pooled"] = pn / pd_
            # P17-5d coverage share vs b(375) across the five SFT arms (not goldlin: same adapters as gold_250)
            arms = [a for a in REF17A + ("rft", "hs", "tunc") if f"{a}|pre" in v.d.cells and f"{o}|375|{a}|b" in out]
            if len(arms) >= 3:
                sh = [unsolved_share(v, o, f"{a}|pre") for a in arms]
                for a, s_ in zip(arms, sh):
                    out[f"{o}|375|{a}|unsolved_share"] = s_
                out[f"{o}|375|ordering|spearman_cov_b"] = spearman(sh, [out[f"{o}|375|{a}|b"] for a in arms])
                out[f"{o}|375|ordering|tunc_minus_teacher"] = out.get(f"{o}|375|tunc|b", np.nan) - out.get(f"{o}|375|teacher_250|b", np.nan)
        return out
    return fn


def decide17a(obs, reps, nulls):
    res = {}
    ci = lambda k: L.ci(reps[k]) if k in reps else (np.nan, np.nan)
    for o in GR:
        P(f"\n## 17A registered predictions [{o}]")
        # P17-1
        d = obs.get(f"{o}|375|clock|delta_tau", np.nan)
        g = obs.get(f"{o}|375|clock|delta_step", np.nan)
        pe = obs.get(f"{o}|clock|probe_err", np.nan)
        held = abs(d) <= REG17["CLOCK_EPS"] and g >= REG17["STEP_GAP"]
        P(f"  P17-1b clock: b_goldlin(375) {fmt(obs.get(f'{o}|375|goldlin|b', np.nan), ci(f'{o}|375|goldlin|b'))}; gold_250 at "
          f"tau {TAU_L[375]:.1f} {fmt(obs.get(f'{o}|375|clock|b_ref_tau', np.nan))}; delta {fmt(d, ci(f'{o}|375|clock|delta_tau'))}; "
          f"minus step-matched gold_250(375): {fmt(g, ci(f'{o}|375|clock|delta_step'))} -> " + ("HELD" if held else "not held"))
        P(f"  P17-1a early probe: forecast {fmt(obs.get(f'{o}|clock|probe_forecast', np.nan))}, error {fmt(pe, ci(f'{o}|clock|probe_err'))} -> "
          + ("HELD" if abs(pe) <= REG17["PROBE_EPS"] else "not held"))
        res[(o, "P17-1")] = held
        # P17-2
        for arm, cmin, bsmin in (("rft", REG17["RFT_C"], REG17["RFT_BS"]), ("hs", REG17["HS_C"], REG17["HS_BS"])):
            pre = f"{o}|375|{arm}|"
            bs, c, vd, b = obs.get(pre + "beta_S", np.nan), obs.get(pre + "c", np.nan), obs.get(pre + "VDpre", np.nan), obs.get(pre + "b", np.nan)
            lo, hi = ci(pre + "beta_S")
            if not (np.isfinite(vd) and vd >= REG17["GATE_VD"] and np.isfinite(b) and 1 - b >= REG17["GATE_1MB"]):
                v_ = "UNTESTABLE (Var_true(D_pre) < 0.004 or 1 - b < 0.15)"
            elif bs <= REG17["UNDO_BS"] and hi < REG17["UNDO_CI"]:
                v_ = "FALSIFIED (undoing: the head-start account is wrong)"
            elif c >= cmin and bs >= bsmin:
                v_ = "HELD (convergence only)"
            else:
                v_ = "inconclusive"
            P(f"  P17-2 {arm}: beta_S {fmt(bs, (lo, hi))}; c {fmt(c, ci(pre + 'c'))}; Var_true(D_pre) {vd:.4f} -> {v_}")
            res[(o, "P17-2", arm)] = v_
        # P17-3
        sp_ = obs.get(f"{o}|hs|semigroup_pooled", np.nan)
        a, b_ = REG17["SEMI"]
        fa, fb = REG17["SEMI_FAIL"]
        v_ = "HELD" if a <= sp_ <= b_ else ("FALSIFIED" if np.isfinite(sp_) and not (fa <= sp_ <= fb) else "not held")
        P(f"  P17-3 semigroup (HS; REPORTED ONLY: not identifiable at these sizes, synthetic slope SD 0.9-6.8), t = 75 and 150 pooled: slope {fmt(sp_, ci(f'{o}|hs|semigroup_pooled'))} -> {v_}; reported per t: "
          + ", ".join(f"{t}: {obs.get(f'{o}|{t}|hs|semigroup', np.nan):+.2f}" for t in (75, 150, 225, 300)))
        # P17-5
        bt = {a: obs.get(f"{o}|375|{a}|b", np.nan) for a in REF17A + ("rft", "hs", "tunc")}
        lo, _ = ci(f"{o}|375|ordering|tunc_minus_teacher")
        largest = all(bt["tunc"] > x for a, x in bt.items() if a != "tunc" and np.isfinite(x))
        P(f"  P17-5b T_unc has the largest b(375): " + ", ".join(f"{a} {x:+.3f}" for a, x in bt.items())
          + f"; T_unc - teacher_250 lower 95% {lo:+.3f} -> " + ("HELD" if largest and lo > 0 else "not held"))
        P(f"  P17-5c RFT and HS b(375) <= 0.15: rft {bt['rft']:+.3f} hs {bt['hs']:+.3f} -> "
          + ("HELD" if bt["rft"] <= REG17["RFTHS_B375"] and bt["hs"] <= REG17["RFTHS_B375"] else "not held") + " (registered bar 0.25)")
        rho = obs.get(f"{o}|375|ordering|spearman_cov_b", np.nan)
        P(f"  P17-5d coverage does not order the arms: Spearman(unsolved share, b(375)) {rho:+.2f} over 5 arms -> "
          + ("HELD" if np.isfinite(rho) and abs(rho) < REG17["COV_RHO"] else "not held (item-wise reachability revived)"))
        # P17-6 (corrected bounds)
        for arm in ARMS17A:
            bs_ = [obs.get(f"{o}|{t}|{arm}|b", np.nan) for t in STEPS17A]
            amp = []
            for t0, t1 in zip(STEPS17A, STEPS17A[1:]):
                k0, k1 = f"{o}|{t0}|{arm}|b", f"{o}|{t1}|{arm}|b"
                if k0 in reps and k1 in reps:
                    inc = obs[k1] - obs[k0]
                    lo_, _ = L.ci(np.asarray(reps[k1]) - np.asarray(reps[k0]))
                    if inc > REG17["AMP"] and lo_ > 0:
                        amp.append(f"{t0}->{t1}")
            nl = nulls.get((o, arm))
            lb = obs.get(f"{o}|375|{arm}|L_b", np.nan)
            over = nl is not None and lb > med(nl["L_b"]) + REG17["AMP"]
            P(f"  P17-6 {arm}: b by step " + " ".join(f"{x:+.2f}" for x in bs_) + f"; logit b(375) {lb:+.3f} vs its null "
              f"{med(nl['L_b']) if nl is not None else np.nan:+.3f} -> " + ("FALSIFIED (amplification " + ",".join(amp) + ")" if amp or over else "HELD"))
        # P17-7 level
        for arm in ("rft", "hs"):
            lr = obs.get(f"{o}|375|{arm}|level_ret", np.nan)
            P(f"  P17-7 {arm}: level retention {fmt(lr, ci(f'{o}|375|{arm}|level_ret'))} (logit gap pre "
              f"{obs.get(f'{o}|375|{arm}|L_gap_pre', np.nan):+.3f}) -> "
              + ("UNTESTABLE (|pre gap| < 0.10)" if not np.isfinite(lr) else "HELD" if lr <= REG17["LEVEL_MAX"] else "not held"))
    return res


def part_17a(paths, args):
    P("\n# 17A — GSM8K new arms (RFT, HS, T_unc, gold-linear) under the 08 S' protocol")
    ev_A, unt, e2u = S.load_A(paths)
    ev = {}
    for f in sorted((paths.root / P17A).glob("*.jsonl")):
        for k, e in L.read_probe(str(f), "gsm8k", key_fn=parse17a).items():
            ev[k] = e
    for k, e in ev_A.items():
        ev.setdefault(k, e)
    bad, rates = set(), {}
    for arm in ARMS17A + ("grpolin",):
        meth = {"rft": "random_sft_250", "hs": "random_sft_250", "tunc": "listed_sft", "goldlin": "random_sft_250", "grpolin": "full_grpo"}[arm]
        for s in SEEDS:
            f = paths.root / f"runs/gsm8k/Llama-3.2-1B-Instruct-p17a-{arm}/seed_{s}/{meth}/grpo/rollouts.jsonl"
            if f.exists():
                rates[(arm, s)] = r = S.stop_rate_rollouts(f)[0]
                if r < S.REG["STOP_MIN"]:
                    bad.add((arm, s))
    bad15, _ = S.part_A_exclusions(paths)
    bad |= {(a, s) for a, s in bad15 if a in REF17A + ("grpo",)}
    P("stop rates:", {f"{a}/{s}": round(r, 3) for (a, s), r in sorted(rates.items())}, "| excluded:", sorted(bad) or "none")
    cells = {"U": L.Cell.from_evals("U", "gsm8k", None, [(0, [unt])] + [(i + 1, [e]) for i, e in enumerate(e2u)])}

    def add(name, lineage, key):
        units = [(s, [ev[(key[0], s, key[1])]]) for s in SEEDS
                 if (key[0], s, key[1]) in ev and (lineage, s) not in bad and (key[0], s) not in bad]
        if len(units) >= 2:
            cells[name] = L.Cell.from_evals(name, "gsm8k", lineage, units)
    for arm in REF17A + ("rft", "hs", "tunc"):
        add(f"{arm}|pre", arm, (arm, "pre"))
    for t in STEPS17A:
        for arm in REF17A + ("rft", "hs", "tunc", "grpo", "grpolin"):
            add(f"{arm}|{t}", arm, (arm, t))
        add(f"goldlin|{t}", "gold_250", ("goldlin", t))  # same SFT adapters as gold_250: one lineage (paired)
    P("cells:", sorted(cells))
    d = L.Design(cells)
    obs, reps = L.bootstrap(d, stats17a_fn(), args.B, seed=1701)
    nulls = {}
    for o in GR:
        aq = {a: q for a, q in quads17a(cells, 375).items() if a in ARMS17A}
        if aq:
            nl, _ = S.run_nulls(d, o, aq, {a: (a,) for a in aq}, args.nsim, S.TAU, seed=1702, kinds=("tilt",))
            nulls.update({(o, a): v for (k, a), v in nl.items()})
    for k in sorted(obs):
        P(f"  {k:46s} {fmt(obs[k], L.ci(reps[k]))}")
    return decide17a(obs, reps, nulls)


# ================================================================ 17B
FAMILIES = {"llama": "unsloth/Llama-3.2-1B-Instruct", "gemma": "google/gemma-3-1b-it"}
ARMS17B = ("gold", "teacher", "rft")
SEEDS17B = {"grpo": (42, 43, 44), "gold": (42, 43), "teacher": (42, 43), "rft": (42, 43)}
STEPS17B = (150, 300)


def run_dir17b(root, fam, proto, arm):
    return root / f"runs/math35d/{fam}-p17b-{proto}-{arm}"


P17B = "runs/probes/prereg17b"
STAGE_FILES17B = {"untrained": "pre", "sft": "pre", "s150": 150, "s300": 300}


def load_17b(paths, fam, proto):
    """All scored 17B evaluations are probe.py outputs (T 0.6, top-p 0.95, 1,024 tokens, 1,000 test items):
    runs/probes/prereg17b/<fam>_<proto>_{untrained (k = 24), sft, s150, s300 (k = 8)}.jsonl, regraded here with the
    pipeline's own math35d grader (sympy; route_sft_grpo.answer_is_correct), lenient and marked."""
    sys.path.insert(0, str(paths.root))
    import route_sft_grpo as R
    rx = re.compile(rf"/math35d/{fam}-p17b-{proto}-(grpo|gold|teacher|rft)/seed_(\d+)/(listed_sft|full_grpo)/(sft|grpo)"
                    r"(?:/checkpoint-(\d+))?/?$")
    evs, agree = {}, []
    for tag in STAGE_FILES17B:
        f = paths.root / P17B / f"{fam}_{proto}_{tag}.jsonl"
        if not f.exists():
            continue
        by = defaultdict(dict)
        for r in L.read_jsonl(f):
            by[r.get("adapter")][(int(r["example_id"]), int(r["sample"]))] = r
        for a, rows in by.items():
            if a is None:
                key = ("U", 0, "pre")
            else:
                m = rx.search(a)
                assert m, f"unparsed 17B adapter {a}"
                arm, seed, _, stage, ck = m.groups()
                key = (arm, int(seed), "pre" if stage == "sft" else int(ck))
            ids = sorted({e for e, _ in rows})
            k = max(s for _, s in rows) + 1
            assert all((e, s) in rows for e in ids for s in range(k)), f"incomplete {a}"
            arr = {o_: np.zeros((len(ids), k), np.uint8) for o_ in L.OUTCOMES}
            for j, e in enumerate(ids):
                for s in range(k):
                    r = rows[(e, s)]
                    ln = R.answer_is_correct(r["completion"], r["gold"], False, "math35d")
                    mk = R.answer_is_correct(r["completion"], r["gold"], True, "math35d")
                    st = bool(r.get("stopped", True))
                    agree.append(mk == bool(r["correct"]))
                    arr["len"][j, s], arr["mk"][j, s], arr["stop"][j, s] = ln, mk, st
                    arr["len_fin"][j, s], arr["mk_fin"][j, s] = ln and st, mk and st
            evs[key] = L.Eval(str(a), arr, k, ids, [rows[(e, 0)]["gold"] for e in ids])
    return evs, (float(np.mean(agree)) if agree else np.nan)


def quads17b(cells, step):
    return {arm: (f"{arm}|pre", "U", f"{arm}|{step}", f"grpo|{step}") for arm in ARMS17B
            if all(c in cells for c in (f"{arm}|pre", "U", f"{arm}|{step}", f"grpo|{step}"))}


def stats17b_fn(graders=GR):
    def fn(v):
        out = {}
        for o in graders:
            for t in STEPS17B:
                for arm, q in quads17b(v.d.cells, t).items():
                    out.update(arm_stats(v, o, q, f"{o}|{t}|{arm}|"))
                ks = {a: f"{o}|{t}|{a}|c" for a in ARMS17B}
                if all(k in out for k in ks.values()):
                    out[f"{o}|{t}|c_teacher_minus_gold"] = out[ks["teacher"]] - out[ks["gold"]]
                    out[f"{o}|{t}|c_teacher_minus_rft"] = out[ks["teacher"]] - out[ks["rft"]]
                lk = {a: f"{o}|{t}|{a}|level_ret" for a in ("teacher", "gold")}
                if all(k in out for k in lk.values()):
                    out[f"{o}|{t}|level_teacher_minus_gold"] = out[lk["teacher"]] - out[lk["gold"]]
        return out
    return fn


def decide17b(obs, reps, nulls, fam):
    res = {}
    ci = lambda k: L.ci(reps[k]) if k in reps else (np.nan, np.nan)
    for o in GR:
        P(f"\n## 17B registered predictions [{fam}, {o}, step 300]")
        for arm in ("gold", "teacher"):
            k = f"{o}|300|{arm}|b"
            nl = nulls.get((o, arm))
            nm = med(nl["b"]) if nl is not None else np.nan
            lo, hi = ci(k)
            held = np.isfinite(nm) and obs.get(k, np.nan) < nm - REG17["B_NULL_GAP"] and hi < nm - REG17["B_NULL_GAP"]
            P(f"  B1 contraction {arm}: b {fmt(obs.get(k, np.nan), (lo, hi))} vs one-index null {nm:+.3f} -> " + ("HELD" if held else "not held"))
            res[(o, "B1", arm)] = held
        c = {a: obs.get(f"{o}|300|{a}|c", np.nan) for a in ARMS17B}
        dtg = obs.get(f"{o}|300|c_teacher_minus_gold", np.nan)
        lo, _ = ci(f"{o}|300|c_teacher_minus_gold")
        r1 = c["gold"] <= REG17["C_GOLD_MAX"] and c["teacher"] >= REG17["C_CONV_MIN"] and c["rft"] >= REG17["C_CONV_MIN"]
        r2 = dtg > 0 and lo > 0 and c["teacher"] <= c["rft"] + REG17["C_TOL"]
        P(f"  B2 routes: c gold {c['gold']:+.3f}, teacher {c['teacher']:+.3f}, rft {c['rft']:+.3f}; c_teacher - c_gold "
          f"{fmt(dtg, (lo, ci(f'{o}|300|c_teacher_minus_gold')[1]))} -> " + ("HELD" if r1 and r2 else "not held")
          + f" (split {'ok' if r1 else 'fails'}; order {'ok' if r2 else 'fails'})")
        res[(o, "B2")] = r1 and r2
        dl = obs.get(f"{o}|300|level_teacher_minus_gold", np.nan)
        lo, hi = ci(f"{o}|300|level_teacher_minus_gold")
        P(f"  B3 level retention teacher - gold: {fmt(dl, (lo, hi))} -> " + ("HELD" if np.isfinite(dl) and dl > REG17["LEVEL_DIFF"] and lo > 0 else "not held"))
        res[(o, "B3")] = np.isfinite(dl) and dl > 0 and lo > 0
    return res


def part_17b(paths, args):
    res = {}
    for fam in FAMILIES:
        P(f"\n# 17B — MATH L3-5, {fam} ({FAMILIES[fam]}), protocol {args.proto}")
        evs, agree = load_17b(paths, fam, args.proto)
        P(f"marked regrade vs stored 'correct': {agree:.5f}")
        if ("U", 0, "pre") not in evs:
            P("untrained evaluation missing: not scored")
            continue
        cells = {"U": L.Cell.from_evals("U", "math35d", None, [(0, [evs[("U", 0, "pre")]])])}
        bad = set()
        for arm in ("grpo",) + ARMS17B:
            meth = "full_grpo" if arm == "grpo" else "listed_sft"
            for s in SEEDS17B[arm]:
                f = run_dir17b(paths.root, fam, args.proto, arm) / f"seed_{s}/{meth}/grpo/rollouts.jsonl"
                if f.exists() and S.stop_rate_rollouts(f)[0] < S.REG["STOP_MIN"]:
                    bad.add((arm, s))
        P("excluded runs (stop rate < 0.5 over the last 20 steps):", sorted(bad) or "none")
        for arm in ("grpo",) + ARMS17B:
            for stage in ("pre",) + STEPS17B:
                units = [(s, [evs[(arm, s, stage)]]) for s in SEEDS17B[arm] if (arm, s, stage) in evs and (arm, s) not in bad]
                if len(units) >= 2:
                    cells[f"{arm}|{stage}"] = L.Cell.from_evals(f"{arm}|{stage}", "math35d", arm, units)
        d = L.Design(cells)
        obs, reps = L.bootstrap(d, stats17b_fn(), args.B, seed=1711)
        nulls = {}
        for o in GR:
            aq = {a: q for a, q in quads17b(cells, 300).items() if a in ("gold", "teacher")}
            if aq:
                nl, _ = S.run_nulls(d, o, aq, {a: (a,) for a in aq}, args.nsim, S.TAU, seed=1712, kinds=("tilt",))
                nulls.update({(o, a): v for (k, a), v in nl.items()})
        for k in sorted(obs):
            P(f"  {k:46s} {fmt(obs[k], L.ci(reps[k]))}")
        res[fam] = decide17b(obs, reps, nulls, fam)
    return res


# ================================================================ schema-only
def schema17(paths, part, proto):
    P(f"# schema-only check, part {part}: file presence and counts; no outcome field is read")
    if part == "17A":
        for f in ("runs/gsm8k/Llama-3.2-1B-Instruct/traces/p17_self_k8.jsonl", "runs/gsm8k/Llama-3.2-1B-Instruct/traces/p17_tunc_ids.json") + \
                tuple(f"runs/gsm8k/Llama-3.2-1B-Instruct/traces/p17_hs_s{s}.jsonl" for s in SEEDS):
            P(f"  {f}: {'exists' if (paths.root / f).exists() else 'MISSING'}")
        for f in sorted((paths.root / P17A).glob("*.jsonl")):
            info, _ = L.schema_probe(str(f))
            keys = [parse17a(a) for a in info]
            P(f"  {f.name}: {len(info)} adapters, rows {[d['rows'] for d in info.values()][:3]}..., parsed {sorted(set((k[0], k[2]) for k in keys), key=str)}")
    else:
        for f in ("data/math35/traces/teacher_llama8b.jsonl", "data/math35/traces/llama1b.jsonl", "data/math35/traces/gemma1b.jsonl") + \
                tuple(f"data/math35/p17b_sft_ids_{a}_{fam}.json" for a in ARMS17B for fam in FAMILIES):
            P(f"  {f}: {'exists' if (paths.root / f).exists() else 'MISSING'}")
        for fam in FAMILIES:
            for arm in ("grpo",) + ARMS17B:
                for s in SEEDS17B[arm]:
                    base = run_dir17b(paths.root, fam, proto, arm) / f"seed_{s}"
                    meth = "full_grpo" if arm == "grpo" else "listed_sft"
                    fs = ([] if arm == "grpo" else [f"{meth}/sft/adapter_model.safetensors"]) + \
                         [f"{meth}/grpo/checkpoint-{t}/adapter_model.safetensors" for t in STEPS17B] + [f"{meth}/grpo/rollouts.jsonl"]
                    P(f"  {fam} {arm} seed {s}: " + ", ".join(f"{Path(f).parent.name}/{Path(f).name} {'ok' if (base / f).exists() else 'MISSING'}" for f in fs))
            for tag in STAGE_FILES17B:
                pf = paths.root / P17B / f"{fam}_{proto}_{tag}.jsonl"
                if pf.exists():
                    info, _ = L.schema_probe(str(pf))
                    P(f"  {pf.name}: {len(info)} adapters, rows {sorted({d['rows'] for d in info.values()})}")
                else:
                    P(f"  {pf.name}: MISSING")


# ================================================================ synthetic AR worlds (dry run)
def _units(rng, name, lineage, probs, k, set_name, seeds):
    U = len(probs)
    C = np.zeros((2, U, len(probs[0])))
    for u, p in enumerate(probs):
        C[0, u] = rng.binomial(k // 2, p)
        C[1, u] = rng.binomial(k - k // 2, p)
    Nh = np.array([[k // 2] * U, [k - k // 2] * U], float)
    return L.Cell(name, set_name, lineage, list(seeds)[:U], {"len": C}, Nh)


def ar_world(rng, N, arms, steps, sched, kappa=0.0066, k=4, nU=4, seeds=SEEDS, rl_seeds=SEEDS, set_name="gsm8k",
             theta=(-0.15, 2.3), clock="cumlr", seed_sd=0.15, extra_rl=None):
    """Attractor model with partial reachability (Def. 2, single rate) on the logit scale:
    l_U = th + e_U; RL alone (seed s): l_R(tau) = l_U + (1 - e^{-kappa tau}) (m + psi + g_s);
    SFT start (seed s): delta_s = delta_r + h_s + u, delta_r = a psi + r (reachable; convergence share
    c = a Var(psi) / Var(delta_r)), h_s the adapter's own seed effect (reachable), u unreachable (frozen; it carries the
    level that is kept); l_S(tau) = l_U + u + d + (1 - e^{-kappa tau}) (m + psi + g'_s - d), d = delta_r + h_s.
    Run effects g_s, g'_s and adapter effects h_s persist across checkpoints (one draw per run).
    arms: {name: dict(c, sd, mu_r, mu_u, sd_u) | dict(kind='hs') (head start: delta_r = (1 - e^{-kappa tau_75})(m + psi))
    | dict(pre_as=other)} (same SFT adapters as `other`, e.g. goldlin = gold_250 under the linear schedule).
    sched: {run name: 'const' | 'lin'}; clock 'step' replaces cumulative lr by the step count."""
    th = rng.normal(theta[0], theta[1], N)
    lU = th + rng.normal(0, 0.8, N)
    psi, m = rng.normal(0, 1.2, N), 1.0
    tau = lambda t, s: (t if clock == "step" else (tau_const(t) if s == "const" else tau_lin(t, max(steps))))
    cells = {"U": _units(rng, "U", None, [L.expit(lU)] * nU, k, set_name, range(nU))}
    cells["U"].lineage = None
    T = {"U": L.expit(lU)}
    rl_runs = {"grpo": sched.get("grpo", "const")}
    rl_runs.update(extra_rl or {})
    for rn, sc in rl_runs.items():
        g = {s: rng.normal(0, seed_sd, N) for s in rl_seeds}
        for t in steps:
            w = 1 - np.exp(-kappa * tau(t, sc))
            ps = [L.expit(lU + w * (m + psi + g[s])) for s in rl_seeds]
            cells[f"{rn}|{t}"] = _units(rng, f"{rn}|{t}", rn, ps, k, set_name, rl_seeds)
            T[f"{rn}|{t}"] = np.mean(ps, 0)
    for name, sp in arms.items():
        pre_name = sp.get("pre_as", name)
        if "pre_as" in sp:  # same SFT adapters as another arm: same pre-RL cell and adapter effects
            dr, u, h = arms[pre_name]["_dru"]
        else:
            if sp.get("kind") == "hs":
                dr = (1 - np.exp(-kappa * tau_const(sp.get("hs_step", 75)))) * (m + psi)
                u = np.zeros(N)
            else:
                c, sd = sp["c"], sp["sd"]
                a = c * sd ** 2 / 1.2 ** 2
                sr = np.sqrt(max(sd ** 2 - a ** 2 * 1.2 ** 2, 1e-6))
                dr = a * psi + rng.normal(sp.get("mu_r", 0.0), sr, N)
                u = rng.normal(sp.get("mu_u", 0.0), sp.get("sd_u", 0.0), N)
            h = {s: rng.normal(0, seed_sd, N) for s in seeds}
            arms[name]["_dru"] = (dr, u, h)
            ps = [L.expit(lU + dr + h[s] + u) for s in seeds]
            cells[f"{name}|pre"] = _units(rng, f"{name}|pre", name, ps, k, set_name, seeds)
            T[f"{name}|pre"] = np.mean(ps, 0)
        g = {s: rng.normal(0, seed_sd, N) for s in seeds}
        sc = sched.get(name, "const")
        for t in steps:
            w = 1 - np.exp(-kappa * tau(t, sc))
            ps = [L.expit(lU + u + dr + h[s] + w * (m + psi + g[s] - dr - h[s])) for s in seeds]
            cells[f"{name}|{t}"] = _units(rng, f"{name}|{t}", pre_name, ps, k, set_name, seeds)
            T[f"{name}|{t}"] = np.mean(ps, 0)
    return L.Design(cells), T


def arms17a_spec(rft_c=1.0, hs="hs"):
    return {"gold_250": dict(c=0.2, sd=1.3, mu_r=-0.7), "teacher_250": dict(c=0.8, sd=0.9, mu_r=0.0, mu_u=0.16, sd_u=0.6),
            "rft": dict(c=rft_c, sd=0.6, mu_r=0.1), "hs": dict(kind=hs) if hs == "hs" else dict(c=0.3, sd=0.8),
            "tunc": dict(c=0.6, sd=0.5, mu_u=0.1, sd_u=1.0), "goldlin": dict(pre_as="gold_250")}


def dry_synthetic17(args):
    from multiprocessing import Pool
    P("# DRY RUN 17 (synthetic AR worlds, single rate on cumulative lr unless stated; Part-A / 17B sizes)")
    worlds = [("AR, cumulative-lr clock (all predictions true)", dict(clock="cumlr")),
              ("step clock (P17-1 false)", dict(clock="step")),
              ("RFT and HS undo (P17-2/3 false)", dict(clock="cumlr", rft_c=0.2, hs="sft"))]
    jobs = [(w, kw, 17000 + 100 * i + r) for i, (w, kw) in enumerate(worlds) for r in range(args.power_reps)]
    with Pool(args.procs) as pool:
        out = pool.map(_w17a, jobs, chunksize=1)
    P("\n## 1. 17A: registered rules in synthetic worlds (lenient; point-estimate parts of each rule; no bootstrap)")
    for w, _ in worlds:
        rr = [x for x in out if x[0] == w]
        keys = sorted(rr[0][1])
        P(f"\n  world: {w} ({len(rr)} datasets)")
        for k in keys:
            vals = np.array([x[1][k] for x in rr], float)
            P(f"    {k:36s} mean {np.nanmean(vals):+.3f} SD {np.nanstd(vals):.3f}" + (f"  P(rule) {np.nanmean(vals > 0.5):.2f}" if k.startswith("rule") else ""))
    jobs = [(w, kw, 17500 + 100 * i + r) for i, (w, kw) in enumerate(W17B) for r in range(args.power_reps)]
    with Pool(args.procs) as pool:
        out = pool.map(_w17b, jobs, chunksize=1)
    P("\n## 2. 17B power (one family; MATH L3-5 sizes: 1,000 items, k = 8, SFT arms 2 seeds, RL alone 3, untrained 3 x 8)")
    for w, _ in W17B:
        rr = [x for x in out if x[0] == w]
        P(f"\n  world: {w} ({len(rr)} datasets)")
        for k in sorted(rr[0][1]):
            vals = np.array([x[1][k] for x in rr], float)
            P(f"    {k:36s} mean {np.nanmean(vals):+.3f} SD {np.nanstd(vals):.3f}" + (f"  P(rule) {np.nanmean(vals > 0.5):.2f}" if k.startswith("rule") else ""))


def _w17a(job):
    w, kw, seed = job
    rng = np.random.default_rng(seed)
    spec = arms17a_spec(kw.get("rft_c", 1.0), kw.get("hs", "hs"))
    sched = {"goldlin": "lin", "grpolin": "lin"}
    d, T = ar_world(rng, 1319, spec, STEPS17A, sched, clock=kw["clock"], extra_rl={"grpolin": "lin"})
    # goldlin's comparator is grpolin
    v = L.View(d)
    o = stats17a_fn(graders=("len",))(v)
    r = {}
    g = lambda k: o.get(f"len|{k}", np.nan)
    r["b375 gold_250"], r["b375 goldlin"], r["clock delta_tau"], r["clock delta_step"] = g("375|gold_250|b"), g("375|goldlin|b"), g("375|clock|delta_tau"), g("375|clock|delta_step")
    r["clock probe_err"] = g("clock|probe_err")
    r["rule P17-1b held"] = float(abs(r["clock delta_tau"]) <= REG17["CLOCK_EPS"] and r["clock delta_step"] >= REG17["STEP_GAP"])
    r["rule P17-1a held"] = float(abs(r["clock probe_err"]) <= REG17["PROBE_EPS"])
    for arm, cm, bm in (("rft", REG17["RFT_C"], REG17["RFT_BS"]), ("hs", REG17["HS_C"], REG17["HS_BS"])):
        r[f"{arm} beta_S"], r[f"{arm} c"], r[f"{arm} VDpre"] = g(f"375|{arm}|beta_S"), g(f"375|{arm}|c"), g(f"375|{arm}|VDpre")
        r[f"{arm} L_beta_S"], r[f"{arm} L_c"] = g(f"375|{arm}|L_beta_S"), g(f"375|{arm}|L_c")
        r[f"rule P17-2 {arm} held"] = float(r[f"{arm} c"] >= cm and r[f"{arm} beta_S"] >= bm)
        r[f"rule P17-2 {arm} undo-point"] = float(r[f"{arm} beta_S"] <= REG17["UNDO_BS"])
    r["hs semigroup 75"], r["hs semigroup 150"], r["hs semigroup pooled"] = g("75|hs|semigroup"), g("150|hs|semigroup"), g("hs|semigroup_pooled")
    r["rule P17-3 held"] = float(REG17["SEMI"][0] <= r["hs semigroup pooled"] <= REG17["SEMI"][1])
    r["rule P17-3 falsified"] = float(np.isfinite(r["hs semigroup pooled"]) and not (REG17["SEMI_FAIL"][0] <= r["hs semigroup pooled"] <= REG17["SEMI_FAIL"][1]))
    for a in ("rft", "hs", "tunc", "teacher_250"):
        r[f"b375 {a}"] = g(f"375|{a}|b")
    r["rule P17-5b largest"] = float(all(r["b375 tunc"] > r[f"b375 {a}"] for a in ("rft", "hs", "teacher_250", "gold_250")))
    r["rule P17-5c rft,hs <= 0.25"] = float(r["b375 rft"] <= REG17["RFTHS_B375"] and r["b375 hs"] <= REG17["RFTHS_B375"])
    r["spearman cov-b"] = g("375|ordering|spearman_cov_b")
    for a in ("rft", "hs"):
        r[f"{a} level_ret"] = g(f"375|{a}|level_ret")
    return w, r


W17B = [("routes as predicted (gold c 0.2, teacher 0.8, RFT 1.0; teacher level kept)", dict(cg=0.2, ct=0.8, cr=1.0, lt=0.3)),
        ("teacher = RFT route (c 0.9 both)", dict(cg=0.2, ct=0.9, cr=0.9, lt=0.3)),
        ("no route difference (all c 0.5; no level)", dict(cg=0.5, ct=0.5, cr=0.5, lt=0.0))]


def _w17b(job):
    w, kw, seed = job
    rng = np.random.default_rng(seed)
    arms = {"gold": dict(c=kw["cg"], sd=1.2, mu_r=-0.5), "teacher": dict(c=kw["ct"], sd=1.0, mu_r=0.2, mu_u=kw["lt"], sd_u=0.3),
            "rft": dict(c=kw["cr"], sd=0.7, mu_r=0.2)}
    d, T = ar_world(rng, 1000, arms, STEPS17B, {}, kappa=0.0066, k=8, nU=3, seeds=(42, 43), rl_seeds=SEEDS,
                    set_name="math35d", theta=(-1.2, 2.3))
    v = L.View(d)
    o = stats17b_fn(graders=("len",))(v)
    g = lambda k: o.get(f"len|300|{k}", np.nan)
    r = {f"b {a}": g(f"{a}|b") for a in ARMS17B}
    r.update({f"c {a}": g(f"{a}|c") for a in ARMS17B})
    r.update({f"L_c {a}": g(f"{a}|L_c") for a in ARMS17B})
    r.update({f"level_ret {a}": g(f"{a}|level_ret") for a in ("gold", "teacher")})
    r["rule B2 split"] = float(r["c gold"] <= REG17["C_GOLD_MAX"] and r["c teacher"] >= REG17["C_CONV_MIN"] and r["c rft"] >= REG17["C_CONV_MIN"])
    r["rule B2 order (point)"] = float(r["c teacher"] > r["c gold"] and r["c teacher"] <= r["c rft"] + REG17["C_TOL"])
    r["rule B3 level (point)"] = float(r["level_ret teacher"] - r["level_ret gold"] > 0)
    return w, r


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["17A", "17B"])
    ap.add_argument("--dry_run", choices=["synthetic"])
    ap.add_argument("--schema_only", action="store_true")
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--proto", choices=["mf", "cont"], default="mf", help="17B protocol variant (the lead picks after Part X)")
    ap.add_argument("--root", default=str(HERE.parents[2]))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--nsim", type=int, default=200)
    ap.add_argument("--power_reps", type=int, default=60)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    paths = S.Paths(args.root, "runs/gsm8k/Llama-3.2-1B-Instruct-px-{proto}{src}")
    P(f"# score_prereg17.py args={vars(args)}\n# REG17={REG17}\n# tau const {TAU_C}\n# tau lin(375) { {k: round(v, 1) for k, v in TAU_L.items()} }")
    t0 = time.time()
    if args.dry_run:
        dry_synthetic17(args)
    elif args.schema_only:
        schema17(paths, args.part, args.proto)
    else:
        if not args.unblind:
            sys.exit("pass --unblind only after prereg 17 is registered and its runs are finished")
        (part_17a if args.part == "17A" else part_17b)(paths, args)
    P(f"\n# done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
