#!/usr/bin/env python3
"""Score prereg/15 (Parts 0, A, B, X, 1) and prereg/16 (OOD retention). Written and dry-run BEFORE registration and
before any per-item analysis of prereg-14 Part A/B/C, prereg-16 or Part X outputs (blinded; EXEC_BRIEF.md section 0).

Usage (Slurm CPU only; never on the login node):
  python runs/theory/scratch/score_prereg15.py --dry_run power              # sign-split calibration -> thresholds
  python runs/theory/scratch/score_prereg15.py --dry_run synthetic          # planted-effect checks, nulls, coverage
  python runs/theory/scratch/score_prereg15.py --dry_run e2                 # reproduce E2 (0.457 / 0.562, R1 bins/pool)
  python runs/theory/scratch/score_prereg15.py --part {0,A,B,X,16} --schema_only   # counts/keys/alignment only
  python runs/theory/scratch/score_prereg15.py --part {0,A,B,X,1,16,all} --unblind # real scoring, after registration

Conventions (registered in prereg/15): lenient grading (extract_final_answer(marked=False)) primary, marked secondary;
probability scale primary, smoothed-logit scale reported; sample halves = samples < k//2 vs the rest of every
evaluation; hierarchical bootstrap = seeds within lineage (an SFT adapter and its RL run are one lineage seed), then
items (GSM-Plus: seed-question clusters; GSM-Symbolic: template clusters), B = 2000; nulls simulated through the
identical cells and estimators (200 replicates) from per-item posterior draws of a 2-D NPMLE prior and ONE link
fitted on the untrained start's own pre/post pair.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p15lib as L  # noqa: E402

SEEDS = (42, 43, 44)
GR = L.GRADERS
STEPS_A = (75, 225, 375)
PRIMARY = 375
ARMS_A = tuple(f"{src}_{n}" for src in ("gold", "teacher") for n in (50, 250, 500))
POOLS = {"gold_pool": ("gold_250", "gold_500"), "teacher_pool": ("teacher_250", "teacher_500")}
GROUPS_A = {**POOLS, **{a: (a,) for a in ARMS_A}}
NULL_KINDS = ("tilt", "affine", "memsym")

# ---------------------------------------------------------------- registered constants (fixed before any analysis)
TAU = 1 / 3          # sign-split selection threshold on the half-sample start difference (power study)
REG = dict(
    # sign-split rule, calibrated in --dry_run power (each synthetic world classified correctly with P >= 0.8)
    # (score_prereg15_dry_power_out.txt, job 26375733: min over 5 symmetric and 3 repair-only worlds of
    # P(correct) = 0.87 at E2-sized effects, tau = 1/3; a fully retained helped effect reads b+ ~ 0.76-0.83)
    SPLIT_D=0.10,     # d = b+(teacher pool) - b-(gold pool): H-sym needs |d| < SPLIT_D, H-repair needs d >= SPLIT_D
    SYM_BP=0.70,      # H-sym also needs b+(teacher pool) <= SYM_BP (helped effects contracted)
    REP_BP=0.50,      # H-repair also needs b+(teacher pool) >= REP_BP
    PURE_REP_BP=0.75,  # 'pure repair' sub-label: b+(teacher pool) >= 0.75 (calibrated reading of full retention)
    MIN_SIDE_N=40, MIN_SIDE_PTS=3.0,          # a side is testable if >= 40 items per arm and chord >= 3 points
    CONTRACT=0.85,                            # 'contraction' = lenient pooled b <= 0.85
    P01_B=0.75, P01_NULL_GAP=0.25, P01_FAIL_B=0.85, P01_FAIL_NULL=0.10,
    POOL_RATIO=0.3, STOP_MIN=0.5,
    X1_MAX=0.15, X_KILL=0.85,
    O1_MIN=0.15, O2_MIN=0.10, O4_MIN_PTS=1.0, GATE16=0.25, HUNIF=0.10,
    K1_MAX=-0.08, K_MARGIN=0.05, K2_MIN=0.10, K3_MIN=0.30,
)


def P(*a):
    print(*a, flush=True)


def fmt(x, ci=None, d=3):
    s = "nan" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"
    if ci is not None:
        lo, hi = ci
        s += " [nan]" if not np.isfinite(lo) else f" [{lo:+.{d}f}, {hi:+.{d}f}]"
    return s


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


# ================================================================ paths and adapter parsing
_RX_G = re.compile(r"/gsm8k/Llama-3.2-1B-Instruct(-marked-steps375|-marked)?(-qwen7b)?/seed_(\d+)/"
                   r"(full_grpo|random_sft_(\d+))/(sft|grpo)(?:/checkpoint-(\d+))?/?$")
_RX_M = re.compile(r"/math/Llama-3.2-1B-Instruct-marked-steps375/seed_(\d+)/(full_grpo|random_sft_250)/grpo"
                   r"(?:/checkpoint-(\d+))?/?$")


def parse_gsm8k_adapter(a):
    """(family, arm, seed, stage): family SFT (after-SFT adapter), S (08 S', 375 steps), C1 (08 C, 1 epoch)."""
    if a is None:
        return ("U", "untrained", None, "pre")
    m = _RX_G.search(a)
    assert m, f"unparsed adapter {a}"
    variant, teacher, seed, method, n, stage, ck = m.groups()
    arm = "grpo" if method == "full_grpo" else f"{'teacher' if teacher else 'gold'}_{n}"
    if stage == "sft":
        return ("SFT", arm, int(seed), "pre")
    return ("C1" if variant == "-marked" else "S", arm, int(seed), int(ck) if ck else "final")


def parse_math_adapter(a):
    m = _RX_M.search(a)
    assert m, f"unparsed adapter {a}"
    seed, method, ck = m.groups()
    return ("grpo" if method == "full_grpo" else "gold_250", int(seed), int(ck) if ck else 375)


class Paths:
    def __init__(self, root, px_template):
        self.root = Path(root)
        r = self.root / "runs"
        self.e2 = str(r / "gsm8k/Llama-3.2-1B-Instruct-censored-")
        self.p14a, self.p14b, self.p14c = r / "probes/prereg14_a", r / "probes/prereg14_b", r / "probes/prereg14_c"
        self.p16 = r / "probes/prereg16"
        self.sprime = {"gold": r / "gsm8k/Llama-3.2-1B-Instruct-marked-steps375",
                       "teacher": r / "gsm8k/Llama-3.2-1B-Instruct-marked-steps375-qwen7b"}
        self.math = r / "math/Llama-3.2-1B-Instruct-marked-steps375"
        self.px = {(src, proto): self.root / px_template.format(proto=proto, src="-qwen7b" if src == "teacher" else "")
                   for src in ("gold", "teacher") for proto in ("cont", "mf")}


# prereg-16 probe outputs (fixed by runs/theory/scratch/submit_prereg16.sh); MATH files are split by source
FILES16 = {
    "gsm_plus": {"untrained": ["gsm_plus_untrained.jsonl"], "sft": ["gsm_plus_sft.jsonl"], "s375": ["gsm_plus_s375.jsonl"]},
    "svamp": {"untrained": ["svamp_untrained.jsonl"], "sft": ["svamp_sft.jsonl"], "s375": ["svamp_s375.jsonl"]},
    "math600": {"untrained": ["math_untrained.jsonl"], "sft": ["math_sft.jsonl", "math_sft_t.jsonl"],
                "s375": ["math_s375.jsonl", "math_s375_t.jsonl"]},
}
DATASET16 = {"gsm_plus": "gsm_plus", "svamp": "svamp", "math600": "math"}
OOD16 = ("gsm_plus", "svamp", "math600")


# ================================================================ run-level diagnostics
def rollouts(path):
    """rollouts.jsonl with resume replays removed (a step number that goes back marks a replay; keep the last pass)."""
    kept = []
    for x in L.read_jsonl(path):
        x = {k: x[k] for k in ("step", "reward", "stopped") if k in x}
        if kept and x["step"] < kept[-1]["step"]:
            kept = [y for y in kept if y["step"] < x["step"]]
        kept.append(x)
    return kept


def stop_rate_rollouts(path, last=20):
    r = rollouts(path)
    smax = max(x["step"] for x in r)
    return float(np.mean([x["stopped"] for x in r if x["step"] > smax - last])), len(r)


def stop_rate_trainer_state(path, last=20):
    """08 S' runs predate --log_rollouts: stop rate = 1 - mean TRL 'completions/clipped_ratio' over the logging entries
    of the last 20 logged steps (entries 360 and 370 = steps 351-370; logging every 10 steps)."""
    lh = [x for x in json.load(open(path))["log_history"] if "completions/clipped_ratio" in x]
    smax = max(x["step"] for x in lh)
    return 1.0 - float(np.mean([x["completions/clipped_ratio"] for x in lh if x["step"] > smax - last]))


def block_reward(path, a, b):
    return float(np.mean([float(x["reward"]) for x in rollouts(path) if a <= x["step"] < b]))


def pool_gap(pairs, first, last):
    """pairs: (SFT-start rollouts, untrained-start rollouts); training-pool reward gap in two step blocks."""
    g0 = [block_reward(s, *first) - block_reward(u, *first) for s, u in pairs]
    g1 = [block_reward(s, *last) - block_reward(u, *last) for s, u in pairs]
    return np.array(g0), np.array(g1)


# ================================================================ generic analysis helpers
def quads(arms, step, prefix="", cells=None):
    q = [(f"{prefix}{a}|pre", f"{prefix}U", f"{prefix}{a}|{step}", f"{prefix}grpo|{step}") for a in arms]
    return [x for x in q if cells is None or all(c in cells for c in x)]


def stats_fn(groups, steps, graders, tau, prefix=""):
    def fn(v):
        out = {}
        for o in graders:
            for st in steps:
                for g, arms in groups.items():
                    q = quads(arms, st, prefix, v.d.cells)
                    if q:
                        out.update(L.all_stats(v, o, q, tau, prefix=f"{o}|{st}|{g}|"))
        return out
    return fn


# ---------------------------------------------------------------- nulls through the identical pipeline
def fit_null_models(d, o, quad):
    S, U, PS, PR = quad
    cU, nU = L.counts(d.cells[U], o)
    cS, nS = L.counts(d.cells[S], o)
    cq, nq = L.counts(d.cells[PS], o)
    cr, nr = L.counts(d.cells[PR], o)
    lat = L.Latent2D(cU, nU, cS, nS)
    m = {"lat": lat, "sU": L.fit_tilt(lat, lat.A, cr, nr), "aff": L.fit_affine(lat, lat.A, cr, nr)}
    m["fS"] = L.fit_link(lat, lat.B, lat.A, cq, nq)
    m["fR"] = L.fit_link(lat, lat.A, lat.B, cr, nr)
    m["lam0"] = 0.5 * (m["fS"]["lam"] + m["fR"]["lam"])
    for k, own, oth, c, n in (("fS0", lat.B, lat.A, cq, nq), ("fR0", lat.A, lat.B, cr, nr)):
        f = m[k[:2]]
        m[k] = L.fit_link(lat, own, oth, c, n, lam=m["lam0"], x0=(f["gamma"], f["beta"], 0.0, f["tau"]))
    return m


def generator(m, kind):
    """tilt: ONE logit-shift link fitted on the untrained start's pair (p_U -> q_R), applied to both starts;
    affine: ONE affine-logit link fitted the same way (sensitivity); memsym: logit q = gamma + beta*own +
    lam0*other + tau*z with the SAME cross-loading lam0 for both starts (no history asymmetry)."""
    if kind == "tilt":
        return lambda rng, a, b: (L.expit(b + m["sU"]), L.expit(a + m["sU"]))
    if kind == "affine":
        g, be = m["aff"]["gamma"], m["aff"]["beta"]
        return lambda rng, a, b: (L.expit(g + be * b), L.expit(g + be * a))
    if kind == "memsym":
        fS, fR, lam = m["fS0"], m["fR0"], m["lam0"]
        return lambda rng, a, b: (
            L.expit(fS["gamma"] + fS["beta"] * b + lam * a + fS["tau"] * rng.standard_normal(len(a))),
            L.expit(fR["gamma"] + fR["beta"] * a + lam * b + fR["tau"] * rng.standard_normal(len(a))))
    raise ValueError(kind)


def sim_parts_once(rng, d, quad, pU, pS, qS, qR, tau):
    S, U, PS, PR = quad
    cells = {U: L.binom_cell(rng, d.cells[U], pU), S: L.binom_cell(rng, d.cells[S], pS),
             PS: L.binom_cell(rng, d.cells[PS], qS), PR: L.binom_cell(rng, d.cells[PR], qR)}
    v = L.View(L.Design(cells))
    return {sc: {"E": L.eiv_parts(v, "sim", *quad, scale=sc), "W": L.wald_parts(v, "sim", *quad, tau, scale=sc),
                 "M": L.memory_parts(v, "sim", *quad, scale=sc), "B": L.bin_parts(v, "sim", *quad, scale=sc)}
            for sc, _ in L.SCALES}


def simulate_parts(rng, d, quad, lat, gen, nsim, tau):
    reps = []
    for _ in range(nsim):
        a, b = lat.draw(rng)
        qS, qR = gen(rng, a, b)
        reps.append(sim_parts_once(rng, d, quad, L.expit(a), L.expit(b), qS, qR, tau))
    return reps


def derive_null(rep_lists):
    """rep_lists: one list of simulated parts per arm (same length); pooled = parts summed over arms per replicate."""
    out = defaultdict(list)
    for parts in zip(*rep_lists):
        for sc, kp in L.SCALES:
            r = {}
            r.update(L.eiv_from(L.add_parts([p[sc]["E"] for p in parts])))
            r.update(L.wald_from(L.add_parts([p[sc]["W"] for p in parts]), len(parts)))
            if sc == "p":
                r.update(L.memory_from(L.add_parts([p[sc]["M"] for p in parts])))
            r.update(L.bins_from(L.add_parts([p[sc]["B"] for p in parts])))
            for k, x in r.items():
                out[kp + k].append(x)
    return {k: np.array(v, float) for k, v in out.items()}


def run_nulls(d, o, arm_quads, groups, nsim, tau, seed, kinds=NULL_KINDS, log=True):
    """Fit per arm, simulate per arm, pool per group. Returns ({(kind, group): {stat: array}}, fitted models)."""
    rng = np.random.default_rng(seed)
    models, sims = {}, {}
    for arm, q in arm_quads.items():
        t0 = time.time()
        models[arm] = m = fit_null_models(d, o, q)
        if log:
            P(f"   null fit [{o}] {arm}: one-index tilt s_U {m['sU']:+.3f}; affine gamma {m['aff']['gamma']:+.3f} "
              f"beta {m['aff']['beta']:+.3f} | latent memory model: free lam_S {m['fS']['lam']:+.3f} lam_R "
              f"{m['fR']['lam']:+.3f} (diff {m['fS']['lam'] - m['fR']['lam']:+.3f}) -> common lam0 {m['lam0']:+.3f} "
              f"({time.time() - t0:.0f}s)")
        for k in kinds:
            sims[(k, arm)] = simulate_parts(rng, d, q, m["lat"], generator(m, k), nsim, tau)
    out = {}
    for k in kinds:
        for g, arms in groups.items():
            if all(a in arm_quads for a in arms):
                out[(k, g)] = derive_null([sims[(k, a)] for a in arms])
    return out, models




# ================================================================ printing and decisions
STAT_ROWS = [("b", "EIV retention b"), ("beta_S", "beta_S (undoing if < 0)"), ("beta_U", "beta_U (convergence if > 0)"),
             ("b_plus", "b+ helped (Wald chord, x-fit)"), ("b_minus", "b- hurt (Wald chord, x-fit)"),
             ("b_plus_minus_b_minus", "b+ - b-"), ("origin_plus", "b+ through origin (Def. 1)"),
             ("origin_minus", "b- through origin (Def. 1)"), ("undo_plus", "1-b+: undoing part"),
             ("conv_plus", "1-b+: convergence part"), ("undo_minus", "1-b-: undoing part"),
             ("conv_minus", "1-b-: convergence part"), ("lam_S", "lam_S (p_U in q_S)"), ("lam_R", "lam_R (p_S in q_R)"),
             ("M", "M = lam_S - lam_R"), ("VDpre", "Var_true(D_pre)"), ("gap_pre_pts", "gap pre (pts)"),
             ("gap_post_pts", "gap post (pts)"), ("rl_alone_pts", "q_R - p_U (pts)"), ("rl_sft_pts", "q_S - p_S (pts)"),
             ("n_plus", "items + per arm"), ("n_minus", "items - per arm"), ("dpre_plus_pts", "chord D_pre + (pts)"),
             ("dpre_minus_pts", "chord D_pre - (pts)")]


def print_group(obs, reps, pre, nulls=None):
    for sc, kp in L.SCALES:
        P(f"   scale {'probability' if sc == 'p' else 'logit (smoothed)'}:")
        for k, lab in STAT_ROWS:
            key = pre + kp + k
            if key not in obs:
                continue
            line = f"    {lab:30s} {fmt(obs[key], L.ci(reps[key]) if reps is not None else None)}"
            for kind, nd in (nulls or {}).items():
                if kp + k in nd and k in ("b", "beta_S", "beta_U", "b_plus", "b_minus", "b_plus_minus_b_minus",
                                          "origin_plus", "origin_minus", "M", "lam_S", "lam_R"):
                    lo, hi = L.ci(nd[kp + k])
                    line += f" | {kind} {med(nd[kp + k]):+.3f} [{lo:+.3f},{hi:+.3f}]"
            P(line)
        tn = (nulls or {}).get("tilt")
        for lab in L.BIN_LABELS:
            k = pre + kp + f"bin_{lab}_ret"
            n = obs.get(pre + f"bin_{lab}_n", 0)
            u = 100 if sc == "p" else 1  # points on the probability scale, logit units otherwise
            line = (f"    bin {lab:4s} n {n:4.0f} D0 {u * obs.get(pre + kp + f'bin_{lab}_D0', np.nan):+6.2f} "
                    f"D1 {u * obs.get(pre + kp + f'bin_{lab}_D1', np.nan):+6.2f} ret {fmt(obs.get(k, np.nan), L.ci(reps[k]) if reps is not None and k in reps else None, 2)}")
            if tn is not None and kp + f"bin_{lab}_ret" in tn:
                m0 = med(tn[kp + f"bin_{lab}_ret"])
                line += f" | tilt null {m0:+.2f} -> excess {obs.get(k, np.nan) - m0:+.2f}"
            P(line)


def bin_excess(obs, reps, nulls, pre, kp=""):
    """Base-success contrast as an EXCESS over the per-bin simulated one-index null (paper Prop. 1d):
    [ret(hi) - null(hi)] - [ret(lo) - null(lo)], bins fixed on the observed data (n >= 15, |D_pre| >= 2 pts)."""
    lab = L.bins_used(obs, pre)
    if lab is None or "tilt" not in nulls:
        return np.nan, (np.nan, np.nan), None
    lo, hi = lab
    tn = nulls["tilt"]
    c = med(tn[kp + f"bin_{hi}_ret"]) - med(tn[kp + f"bin_{lo}_ret"])
    k_hi, k_lo = pre + kp + f"bin_{hi}_ret", pre + kp + f"bin_{lo}_ret"
    val = obs[k_hi] - obs[k_lo] - c
    r = np.asarray(reps[k_hi]) - np.asarray(reps[k_lo]) - c
    return val, L.ci(r), lab


def classify_split(bp, bm):
    d = bp - bm
    if bp <= REG["SYM_BP"] and abs(d) < REG["SPLIT_D"]:
        return "H-sym"
    if bp >= REG["REP_BP"] and d >= REG["SPLIT_D"]:
        return "H-repair (pure)" if bp >= REG["PURE_REP_BP"] else "H-repair"
    return "intermediate"


def decide_split(obs, reps, o, step, gT, gG, label):
    pre_T, pre_G = f"{o}|{step}|{gT}|", f"{o}|{step}|{gG}|"
    bp, bm = obs.get(pre_T + "b_plus", np.nan), obs.get(pre_G + "b_minus", np.nan)
    testable = (obs.get(pre_T + "n_plus", 0) >= REG["MIN_SIDE_N"] and abs(obs.get(pre_T + "dpre_plus_pts", 0)) >= REG["MIN_SIDE_PTS"]
                and obs.get(pre_G + "n_minus", 0) >= REG["MIN_SIDE_N"] and abs(obs.get(pre_G + "dpre_minus_pts", 0)) >= REG["MIN_SIDE_PTS"])
    verdict = classify_split(bp, bm) if testable and np.isfinite(bp) and np.isfinite(bm) else \
        "UNTESTABLE (a side has < 40 items per arm or a chord < 3 points)"
    dd = np.asarray(reps[pre_T + "b_plus"]) - np.asarray(reps[pre_G + "b_minus"])
    P(f"  {label} [{o}, step {step}]: b+({gT}) {fmt(bp, L.ci(reps[pre_T + 'b_plus']))}; b-({gG}) "
      f"{fmt(bm, L.ci(reps[pre_G + 'b_minus']))}; d {fmt(bp - bm, L.ci(dd))} -> {verdict}")
    return verdict


def decide_memory(obs, reps, nulls, o, step, groups):
    res = {}
    for g in groups:
        key = f"{o}|{step}|{g}|M"
        nd = nulls.get(("memsym", g))
        if key not in obs or nd is None:
            continue
        m0 = med(nd["M"])
        lo, hi = L.ci(np.asarray(reps[key]) - m0)
        nlo, nhi = L.ci(nd["M"])
        outside = obs[key] > nhi or obs[key] < nlo
        if lo > 0 and outside:
            v = "UNDOING toward the base profile (asymmetry above the symmetric null)"
        elif hi < 0 and outside:
            v = "CONVERGENCE toward the SFT profile (asymmetry below the symmetric null)"
        else:
            v = "no history asymmetry detected"
        P(f"  memory [{o}, step {step}, {g}]: M {obs[key]:+.3f}; symmetric-null median {m0:+.3f} (95% {nlo:+.3f}..{nhi:+.3f}); "
          f"M - null {obs[key] - m0:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}")
        res[g] = v
    return res


def world_q(kind, par, a, b, sU):
    """Worlds built on this data's latent locations (per-item posterior draws a = logit p_U, b = logit p_S) and the
    one-index RL-alone link q_R = expit(a + s_U). 'sym_p': q_S = q_R + par (p_S - p_U) (probability-symmetric);
    'sym_l': logit q_S = logit q_R + par (b - a) (logit-symmetric); 'repair': q_S = q_R + D for D > 0, par D for D < 0."""
    pU, pS, qR = L.expit(a), L.expit(b), L.expit(a + sU)
    D = pS - pU
    if kind == "sym_p":
        qS = np.clip(qR + par * D, 0, 1)
    elif kind == "sym_l":
        qS = L.expit(a + sU + par * (b - a))
    else:
        qS = np.clip(qR + np.where(D > 0, D, par * D), 0, 1)
    return pU, pS, qS, qR


def sim_world(rng, d, models, quads_map, arms, kind, par, nsim, tau):
    """Pooled statistics (arrays over nsim) of a world passed through the identical cells and estimators."""
    per_arm = []
    for arm in arms:
        m = models[arm]
        reps = []
        for _ in range(nsim):
            a, b = m["lat"].draw(rng)
            reps.append(sim_parts_once(rng, d, quads_map[arm], *world_q(kind, par, a, b, m["sU"]), tau))
        per_arm.append(reps)
    return derive_null(per_arm)


def match_par(rng, d, models, quads_map, arms, kind, target, tau, nsim=30, lo=0.0, hi=1.5, iters=7):
    """Bisection on the world parameter so that the median pooled probability-scale b matches the observed b."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if med(sim_world(rng, d, models, quads_map, arms, kind, mid, nsim, tau)["b"]) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def split_location_check(d, o, models, quads_map, obs, step, gT, gG, armsT, armsG, verdict0, nsim, tau, seed, log=True):
    """Registered location check of the A2/X2 verdict (it can only DOWNGRADE a verdict to 'intermediate').
    Helped (teacher) and hurt (gold) items sit at different base-success locations, so ceiling and floor effects can
    make b+(T) - b-(G) non-zero under symmetric reversion (seen in the mock end-to-end test: a logit-symmetric world
    gave d = +0.15). Worlds are built from THIS data's latent fits: symmetric on the probability scale (sym_p) and on
    the logit scale (sym_l), each with one parameter common to both sources and matched to the mean observed b of the
    two groups, and repair-only (helped kept, hurt retained r with r matched to the gold group's b).
    H-repair stands only if d_obs > the 97.5th percentile of d under BOTH symmetric worlds; H-sym stands only if
    d_obs < the 2.5th percentile of d under the repair world. Also prints the low-power label (fixed rule's
    correct-classification rate < 0.8 in sym_p or repair)."""
    rng = np.random.default_rng(seed)
    bp, bm = obs.get(f"{o}|{step}|{gT}|b_plus", np.nan), obs.get(f"{o}|{step}|{gG}|b_minus", np.nan)
    d_obs = bp - bm
    target = float(np.nanmean([obs.get(f"{o}|{step}|{gT}|b", np.nan), obs.get(f"{o}|{step}|{gG}|b", np.nan)]))
    allarms = list(armsT) + list(armsG)
    q = {}
    pars = {"sym_p": match_par(rng, d, models, quads_map, allarms, "sym_p", target, tau),
            "sym_l": match_par(rng, d, models, quads_map, allarms, "sym_l", target, tau),
            "repair": match_par(rng, d, models, quads_map, list(armsG), "repair", obs.get(f"{o}|{step}|{gG}|b", np.nan), tau)}
    low = False
    for kind, par in pars.items():
        nT = sim_world(rng, d, models, quads_map, armsT, kind, par, nsim, tau)
        nG = sim_world(rng, d, models, quads_map, armsG, kind, par, nsim, tau)
        dd = nT["b_plus"] - nG["b_minus"]
        dd = dd[np.isfinite(dd)]
        q[kind] = np.percentile(dd, [2.5, 50, 97.5])
        cls = [classify_split(x, y) for x, y in zip(nT["b_plus"], nG["b_minus"])]
        ps, pr = np.mean([c == "H-sym" for c in cls]), np.mean([c.startswith("H-repair") for c in cls])
        if kind in ("sym_p", "repair"):
            low |= (ps if kind == "sym_p" else pr) < 0.8
        if log:
            P(f"  location check [{o}] world {kind} (parameter {par:.3f}): d median {q[kind][1]:+.3f} [{q[kind][0]:+.3f}, "
          f"{q[kind][2]:+.3f}]; fixed rule gives P(H-sym) {ps:.2f} P(H-repair) {pr:.2f}")
    v = verdict0
    if verdict0.startswith("H-repair") and not (d_obs > q["sym_p"][2] and d_obs > q["sym_l"][2]):
        v = "intermediate (the asymmetry is within what symmetric reversion produces at these locations)"
    elif verdict0 == "H-sym" and not d_obs < q["repair"][0]:
        v = "intermediate (repair-only not excluded at these locations)"
    if log:
        P(f"  -> final verdict [{o}]: {v} (fixed-threshold verdict: {verdict0}; d_obs {d_obs:+.3f})"
      + ("; LOW-POWER (a calibration world is classified correctly < 80%)" if low else "; at registered power"))
    return v


# ================================================================ Part A (GSM8K, prereg-14 Part A)
def load_A(paths):
    ev = {}
    for f in ("sft", "s75", "s225", "s375"):
        for k, e in L.read_probe(str(paths.p14a / f"{f}.jsonl"), "gsm8k", key_fn=parse_gsm8k_adapter).items():
            ev[(k[1], k[2], k[3])] = e
    unt = L.read_probe(str(paths.p14a / "untrained.jsonl"), "gsm8k")[None]
    e2u = [L.check_aligned(L.read_job(f"{paths.e2}grpo/seed_{s}/base_predictions.jsonl", "gsm8k"), unt.ids, unt.gold,
                           f"E2 base seed {s}") for s in SEEDS]
    return ev, unt, e2u


def part_A_exclusions(paths):
    bad, rates = set(), {}
    for arm in ARMS_A + ("grpo",):
        for s in SEEDS:
            src = "teacher" if arm.startswith("teacher") else "gold"
            method = "full_grpo" if arm == "grpo" else f"random_sft_{arm.split('_')[1]}"
            f = paths.sprime[src] / f"seed_{s}/{method}/grpo/checkpoint-375/trainer_state.json"
            rates[(arm, s)] = r = stop_rate_trainer_state(f)
            if r < REG["STOP_MIN"]:
                bad.add((arm, s))
    return bad, rates


def cells_A(ev, unt, e2u, bad, steps=STEPS_A, prefix="", set_name="gsm8k", u_pool=True, arms=ARMS_A):
    """U = untrained (probe 4 samples + E2 training-job 12 samples: same model, prompt, T 0.6, top-p 0.95, top-k 0,
    512 tokens, data_seed 1234); '<arm>|pre' SFT adapters; '<arm>|<step>', 'grpo|<step>' RL checkpoints. A run excluded
    by its stop rate leaves its whole lineage (SFT adapter + RL run) for SFT arms, the RL cells for GRPO alone."""
    cells = {}
    units = [(0, [unt])] + ([(i + 1, [e]) for i, e in enumerate(e2u)] if u_pool else [])
    cells[prefix + "U"] = L.Cell.from_evals(prefix + "U", set_name, None, units)
    for arm in arms:
        seeds = [s for s in SEEDS if (arm, s) not in bad]
        cells[f"{prefix}{arm}|pre"] = L.Cell.from_evals(f"{prefix}{arm}|pre", set_name, arm,
                                                        [(s, [ev[(arm, s, "pre")]]) for s in seeds])
        for st in steps:
            cells[f"{prefix}{arm}|{st}"] = L.Cell.from_evals(f"{prefix}{arm}|{st}", set_name, arm,
                                                             [(s, [ev[(arm, s, st)]]) for s in seeds])
    for st in steps:
        seeds = [s for s in SEEDS if ("grpo", s) not in bad]
        cells[f"{prefix}grpo|{st}"] = L.Cell.from_evals(f"{prefix}grpo|{st}", set_name, "grpo",
                                                        [(s, [ev[("grpo", s, st)]]) for s in seeds])
    return cells


def part_A(paths, args):
    P("\n# Part A — GSM8K test (prereg-14 Part A files): help vs damage, decomposition, history asymmetry")
    ev, unt, e2u = load_A(paths)
    P(f"marked regrade vs stored 'correct': {L.AGREE[True]}/{sum(L.AGREE.values())} agree")
    assert L.AGREE[False] == 0, "grader port disagrees with the stored marked field"
    bad, rates = part_A_exclusions(paths)
    P("training stop rate, last 20 logged steps:", {f"{a}/{s}": round(r, 3) for (a, s), r in sorted(rates.items())})
    P("excluded runs:", sorted(bad) or "none")
    d = L.Design(cells_A(ev, unt, e2u, bad))
    short = [c.name for c in d.cells.values() if c.lineage is not None and len(c.seeds) < 2]
    if short:
        P("cells with < 2 seeds (their groups are not scored; rerun with new seeds, as in 13):", short)
    obs, reps = L.bootstrap(d, stats_fn(GROUPS_A, STEPS_A, GR, args.tau), args.B, seed=1501)
    nulls, models = {}, {}
    aq = {a: quads([a], PRIMARY)[0] for a in ARMS_A if quads([a], PRIMARY, cells=d.cells)}
    for o in GR:
        nulls[o], models[o] = run_nulls(d, o, aq, GROUPS_A, args.nsim, args.tau, seed=1502)
    for o in GR:
        for st in STEPS_A:
            for g in GROUPS_A:
                pre = f"{o}|{st}|{g}|"
                if pre + "b" in obs:
                    P(f"\n  [{o}] step {st} {g}")
                    print_group(obs, reps, pre, {k: v for (k, gg), v in nulls[o].items() if gg == g} if st == PRIMARY else None)
    P("\n## Part A registered outcomes (step 375; lenient primary, marked secondary)")
    res = {}
    for o in GR:
        res[o] = decide_split(obs, reps, o, PRIMARY, "teacher_pool", "gold_pool", "A2 help vs damage")
        for g in POOLS:
            nd = nulls[o][("tilt", g)]
            b, (lo, hi) = obs[f"{o}|{PRIMARY}|{g}|b"], L.ci(reps[f"{o}|{PRIMARY}|{g}|b"])
            P(f"  A1 contraction [{o}] {g}: b {fmt(b, (lo, hi))}; one-index null median {med(nd['b']):+.3f}; "
              f"logit b {fmt(obs[f'{o}|{PRIMARY}|{g}|L_b'], L.ci(reps[f'{o}|{PRIMARY}|{g}|L_b']))} vs null "
              f"{med(nd['L_b']):+.3f} -> " + ("contraction beyond the null" if hi < med(nd["b"]) - 0.10 else "not beyond the null"))
            val, cint, lab = bin_excess(obs, reps, {k: v for (k, gg), v in nulls[o].items() if gg == g}, f"{o}|{PRIMARY}|{g}|")
            P(f"  A3 base-success excess [{o}] {g} (bins {lab}): {fmt(val, cint)}")
        decide_memory(obs, reps, nulls[o], o, PRIMARY, POOLS)
        if res[o].startswith("H-"):
            res[o] = split_location_check(d, o, models[o], aq, obs, PRIMARY, "teacher_pool", "gold_pool", POOLS["teacher_pool"],
                                          POOLS["gold_pool"], res[o], min(args.nsim, 200), args.tau, seed=1503)
    d4 = L.Design(cells_A(ev, unt, e2u, bad, u_pool=False))
    o4 = stats_fn(POOLS, (PRIMARY,), ("len",), args.tau)(L.View(d4))
    P("  sensitivity, untrained = probe only (4 samples): " + "; ".join(
        f"{g} b {o4[f'len|{PRIMARY}|{g}|b']:+.3f} b+ {o4[f'len|{PRIMARY}|{g}|b_plus']:+.3f} b- "
        f"{o4[f'len|{PRIMARY}|{g}|b_minus']:+.3f}" for g in POOLS))
    return {"verdict": res["len"], "min_b": min(obs.get(f"len|{PRIMARY}|{g}|b", np.inf) for g in POOLS)}


# ================================================================ Part 0 (MATH, prereg-14 Part C)
def load_C(paths):
    ck = L.read_probe(str(paths.p14c / "ckpt.jsonl"), "math", key_fn=parse_math_adapter)
    ref = next(iter(ck.values()))
    al = lambda f, lab: L.check_aligned(L.read_job(str(f), "math"), ref.ids, ref.gold, lab)
    base = [al(paths.math / f"seed_{s}/base_predictions.jsonl", f"base {s}") for s in SEEDS]
    sft = {s: al(paths.math / f"seed_{s}/random_sft_250/predictions_after_sft.jsonl", f"after-SFT {s}") for s in SEEDS}
    job = {(arm, s): al(paths.math / f"seed_{s}/{meth}/predictions.jsonl", f"job {arm} {s}")
           for arm, meth in (("grpo", "full_grpo"), ("gold_250", "random_sft_250")) for s in SEEDS}
    return ck, base, sft, job


def part_0(paths, args):
    P("\n# Part 0 — MATH L1-3 numeric (prereg-14 Part C), per-item contraction (registered after the data existed)")
    ck, base, sft, job = load_C(paths)
    rates, bad = {}, set()
    for arm, meth in (("grpo", "full_grpo"), ("gold_250", "random_sft_250")):
        for s in SEEDS:
            rates[(arm, s)] = r = stop_rate_rollouts(paths.math / f"seed_{s}/{meth}/grpo/rollouts.jsonl")[0]
            if r < REG["STOP_MIN"]:
                bad.add((arm, s))
    P("GRPO stop rate, last 20 steps:", {f"{a}/{s}": round(r, 3) for (a, s), r in sorted(rates.items())},
      "| excluded:", sorted(bad) or "none")
    gs = [s for s in SEEDS if ("gold_250", s) not in bad]
    rs = [s for s in SEEDS if ("grpo", s) not in bad]
    if len(gs) < 2 or len(rs) < 2:
        P("fewer than 2 runs left in a cell -> Part 0 not scored (rerun with new seeds, as in 13)")
        return {}
    cells = {"U": L.Cell.from_evals("U", "math", None, [(i, [e]) for i, e in enumerate(base)]),
             "gold_250|pre": L.Cell.from_evals("gold_250|pre", "math", "gold_250", [(s, [sft[s]]) for s in gs])}
    steps = (75, 225, 375, "375job")
    for st in steps:
        for arm, ss in (("gold_250", gs), ("grpo", rs)):
            cells[f"{arm}|{st}"] = L.Cell.from_evals(f"{arm}|{st}", "math", arm,
                                                     [(s, [job[(arm, s)] if st == "375job" else ck[(arm, s, st)]]) for s in ss])
    d = L.Design(cells)
    G = {"gold_250": ("gold_250",)}
    obs, reps = L.bootstrap(d, stats_fn(G, steps, GR, args.tau), args.B, seed=1504)
    nulls = {}
    for o in GR:
        for st in (375, "375job"):
            nulls[(o, st)], _ = run_nulls(d, o, {"gold_250": quads(["gold_250"], st)[0]}, G, args.nsim, args.tau, seed=1505)
    for o in GR:
        for st in steps:
            P(f"\n  [{o}] step {st} (375job = training-job predictions at 375: independent replication set)")
            print_group(obs, reps, f"{o}|{st}|gold_250|", {k: v for (k, g), v in nulls[(o, st)].items()} if (o, st) in nulls else None)
    P("\n## Part 0 registered decision rules (lenient, probability scale, probe checkpoints at 375)")
    nl = {k: v for (k, g), v in nulls[("len", 375)].items()}
    b, nm = obs["len|375|gold_250|b"], med(nl["tilt"]["b"])
    if b <= REG["P01_B"] and b <= nm - REG["P01_NULL_GAP"]:
        v = "P0-1: CONTRACTION REPLICATES on MATH"
    elif b > REG["P01_FAIL_B"] or b > nm - REG["P01_FAIL_NULL"]:
        v = "P0-1: DOES NOT REPLICATE (stop the reversion line after Part X; measurement-paper fallback)"
    else:
        v = "P0-1: intermediate (neither rule fires)"
    P(f"  P0-1: b {fmt(b, L.ci(reps['len|375|gold_250|b']))}; one-index null median {nm:+.3f} -> {v}")
    P(f"  logit scale (reported): b {fmt(obs['len|375|gold_250|L_b'], L.ci(reps['len|375|gold_250|L_b']))}; null {med(nl['tilt']['L_b']):+.3f}")
    P(f"  replication (training-job predictions at 375): b {fmt(obs['len|375job|gold_250|b'], L.ci(reps['len|375job|gold_250|b']))}")
    val, cint, lab = bin_excess(obs, reps, nl, "len|375|gold_250|")
    P(f"  P0-2 (excess over the per-bin one-index null, bins {lab}): {fmt(val, cint)} -> "
      + ("HOLDS (reversion concentrated on high-base-success items beyond saturation)" if val < 0 and cint[1] < 0 else "not held"))
    vl, cl, _ = bin_excess(obs, reps, nl, "len|375|gold_250|", kp="L_")
    P(f"  P0-2 logit scale (reported): {fmt(vl, cl)}")
    pairs = [(paths.math / f"seed_{s}/random_sft_250/grpo/rollouts.jsonl", paths.math / f"seed_{s}/full_grpo/grpo/rollouts.jsonl")
             for s in SEEDS if s in gs and s in rs]
    g0, g1 = pool_gap(pairs, (0, 25), (350, 375))
    ratio = g1.mean() / g0.mean()
    P(f"  pool equalisation (reported): gap [0,25) {g0.mean():+.3f} -> [350,375) {g1.mean():+.3f}; ratio {ratio:+.2f} -> "
      + ("equalisation assumption FAILS (> 0.3)" if ratio > REG["POOL_RATIO"] else "<= 0.3"))
    return {"b": b, "verdict": v}


# ================================================================ Part X (protocol 2x2, prospective GPU part)
PX_ARMS = ("gold_cont", "gold_mf", "teacher_cont", "teacher_mf")
# RL-alone comparators: E2 = the E2 untrained `grpo` runs (fresh LoRA drawn by TRL inside GRPOTrainer; censor acted
# on <= 3 of 9,600 rollouts); MF = the Part X `--merge_sft_start` full_grpo runs (fresh LoRA from fresh_lora with
# --lora_init_seed S: the same A as the mf arms of seed S). Primary: cont arms vs E2, mf arms vs MF.
PX_PRIMARY = {"cont": "E2", "mf": "MF"}


def px_quads(cells):
    """{group: [quad]}: primary groups named by arm (gold_mf, ...), secondary groups '<arm>@<R>' for both comparators."""
    g = {}
    for arm in PX_ARMS:
        proto = arm.split("_")[1]
        for R in ("E2", "MF"):
            q = (f"{arm}|pre", "U", f"{arm}|150", f"grpo{R}|150")
            if all(c in cells for c in q):
                g[arm if PX_PRIMARY[proto] == R else f"{arm}@{R}"] = [q]
    return g


def part_X(paths, args):
    P("\n# Part X — protocol: continuing adapter vs merged SFT + fresh LoRA (E2 settings, 150 steps, censor off)")
    U = [L.read_job(f"{paths.e2}grpo/seed_{s}/base_predictions.jsonl", "gsm8k") for s in SEEDS]
    cells = {"U": L.Cell.from_evals("U", "gsm8k", None, [(i, [e]) for i, e in enumerate(U)])}
    rates, bad = {}, set()
    runs = {("grpoE2", s): f"{paths.e2}grpo/seed_{s}/full_grpo" for s in SEEDS}
    runs.update({("grpoMF", s): str(paths.px[("gold", "mf")] / f"seed_{s}/full_grpo") for s in SEEDS})
    for arm in PX_ARMS:
        src, proto = arm.split("_")
        runs.update({(arm, s): str(paths.px[(src, proto)] / f"seed_{s}/random_sft_250") for s in SEEDS})
    for (arm, s), base in runs.items():
        rates[(arm, s)] = r = stop_rate_rollouts(f"{base}/grpo/rollouts.jsonl")[0]
        if r < REG["STOP_MIN"]:
            bad.add((arm, s))
    P("GRPO stop rate, last 20 steps:", {f"{a}/{s}": round(r, 3) for (a, s), r in sorted(rates.items())}, "| excluded:", sorted(bad) or "none")
    for R in ("grpoE2", "grpoMF"):
        cells[f"{R}|150"] = L.Cell.from_evals(f"{R}|150", "gsm8k", R, [(s, [L.read_job(f"{runs[(R, s)]}/predictions.jsonl", "gsm8k")])
                                                                        for s in SEEDS if (R, s) not in bad])
    for arm in PX_ARMS:
        src = arm.split("_")[0]
        ss = [s for s in SEEDS if (arm, s) not in bad]
        # lineage = source: the cont and mf arms of a source and seed share the SFT adapter (paired resampling)
        cells[f"{arm}|pre"] = L.Cell.from_evals(f"{arm}|pre", "gsm8k", src, [(s, [L.read_job(f"{runs[(arm, s)]}/predictions_after_sft.jsonl", "gsm8k")]) for s in ss])
        cells[f"{arm}|150"] = L.Cell.from_evals(f"{arm}|150", "gsm8k", src, [(s, [L.read_job(f"{runs[(arm, s)]}/predictions.jsonl", "gsm8k")]) for s in ss])
    short = [c.name for c in cells.values() if c.lineage is not None and len(c.seeds) < 2]
    if short:
        P("cells with < 2 seeds (not scored; rerun with new seeds, as in 13):", short)
    d = L.Design(cells)
    G = px_quads(cells)

    def fn(v):
        out = {}
        for o in GR:
            for g, qq in G.items():
                out.update(L.all_stats(v, o, qq, args.tau, prefix=f"{o}|150|{g}|"))
            for src in ("gold", "teacher"):
                for kp in ("", "L_"):
                    a, c = f"{o}|150|{src}_mf|{kp}b", f"{o}|150|{src}_cont|{kp}b"
                    if a in out and c in out:
                        out[f"{o}|X1|{src}|{kp}delta"] = out[a] - out[c]  # each arm against its primary comparator
                    for R in ("E2", "MF"):  # secondary: both arms against the same comparator
                        a2 = f"{o}|150|{src}_mf{'' if R == 'MF' else '@E2'}|{kp}b"
                        c2 = f"{o}|150|{src}_cont{'' if R == 'E2' else '@MF'}|{kp}b"
                        if a2 in out and c2 in out:
                            out[f"{o}|X1|{src}|{kp}delta_same_{R}"] = out[a2] - out[c2]
        return out
    obs, reps = L.bootstrap(d, fn, args.B, seed=1506)
    nulls, models_x = {}, {}
    prim = {g: qq[0] for g, qq in G.items() if "@" not in g}
    for o in GR:
        nulls[o], models_x[o] = run_nulls(d, o, prim, {g: (g,) for g in prim}, args.nsim, args.tau, seed=1507)
        for g in G:
            P(f"\n  [{o}] {g} step 150 (comparator {G[g][0][3]})")
            print_group(obs, reps, f"{o}|150|{g}|", {k: v for (k, gg), v in nulls[o].items() if gg == g})
    P("\n## Part X registered outcomes")
    res = {}
    for o in GR:
        for src in ("gold", "teacher"):
            k = f"{o}|X1|{src}|delta"
            if k not in obs:
                P(f"  X1 [{o}] {src}: not scored (missing cells)")
                continue
            lo, hi = L.ci(reps[k])
            bmf = obs[f"{o}|150|{src}_mf|b"]
            held = abs(obs[k]) < REG["X1_MAX"]
            P(f"  X1 [{o}] {src}: b(mf vs MF) {fmt(bmf, L.ci(reps[f'{o}|150|{src}_mf|b']))} - b(cont vs E2) "
              f"{fmt(obs[f'{o}|150|{src}_cont|b'], L.ci(reps[f'{o}|150|{src}_cont|b']))} = {fmt(obs[k], (lo, hi))} -> "
              + ("HOLDS (|delta| < 0.15)" if held else "FAILS") + ("; CI inside +-0.15" if lo > -0.15 and hi < 0.15 else "")
              + (f"; KILL: merged-fresh b >= 0.85 -> the {src} contraction is a parameterisation artefact" if bmf >= REG["X_KILL"] else ""))
            for R in ("E2", "MF"):
                k2 = f"{o}|X1|{src}|delta_same_{R}"
                if k2 in obs:
                    P(f"     secondary, both arms vs {R}: delta {fmt(obs[k2], L.ci(reps[k2]))}")
            res[(o, src)] = (held, bmf >= REG["X_KILL"])
        for proto, lab in (("mf", "X2 help vs damage, merged-fresh (primary)"), ("cont", "X2 help vs damage, continuing (secondary)")):
            if f"{o}|150|teacher_{proto}|b_plus" not in obs:
                continue
            v0 = decide_split(obs, reps, o, 150, f"teacher_{proto}", f"gold_{proto}", lab)
            if v0.startswith("H-"):
                split_location_check(d, o, models_x[o], prim, obs, 150, f"teacher_{proto}", f"gold_{proto}", [f"teacher_{proto}"],
                                     [f"gold_{proto}"], v0, min(args.nsim, 200), args.tau, seed=1508)
    x1 = all(res.get(("len", s), (False, False))[0] for s in ("gold", "teacher"))
    kill = all(res.get(("len", s), (False, False))[1] for s in ("gold", "teacher"))
    P(f"  X1 overall (lenient): {'HOLDS' if x1 else 'FAILS'}; headline kill (both sources b_mf >= 0.85): {'YES' if kill else 'no'}")
    P("  Part 1 may run only if X1 holds (lenient) and the kill rule does not fire.")
    return {"X1": x1 and not kill, "kill": kill}


# ================================================================ Part B (GSM-Symbolic, exploratory)
def gsm8k_gold(answer_text):
    """route_sft_grpo.extract_gsm8k_gold."""
    if "####" in answer_text:
        return answer_text.rsplit("####", 1)[-1].strip()
    return L.extract_final_answer(answer_text) or answer_text.strip()


def hf_rows(name, config, split, filt=None):
    """Rows of a cached HF dataset in prepare_data's order: (filter) then shuffle(seed=1234)."""
    import os
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    if filt is not None:
        ds = ds.filter(filt)
    return ds.shuffle(seed=1234)


def part_B(paths, args):
    P("\n# Part B — GSM-Symbolic (prereg-14 Part B files): aggregate arm - GRPO alone (exploratory, reported only)")
    ev = {}
    for f in ("c1ep", "s225", "s375"):
        ev.update(L.read_probe(str(paths.p14b / f"{f}.jsonl"), "gsm_symbolic", key_fn=parse_gsm8k_adapter))
    unt = L.read_probe(str(paths.p14b / "untrained.jsonl"), "gsm_symbolic")[None]
    strata = None
    try:
        ds = hf_rows("apple/GSM-Symbolic", "main", "test")
        assert all(gsm8k_gold(ds[int(e)]["answer"]) == g for e, g in zip(unt.ids, unt.gold)), "order mismatch"
        tmpl = defaultdict(list)
        for j, e in enumerate(unt.ids):
            tmpl[ds[int(e)]["original_id"]].append(j)
        strata = [("clusters", [{"gsm_symbolic": np.array(v)} for v in tmpl.values()])]
        P(f"template clusters: {len(tmpl)}")
    except Exception as e:  # reported-only part: fall back to the item bootstrap
        P("template map unavailable, item bootstrap:", repr(e))
    bad, _ = part_A_exclusions(paths)
    units = defaultdict(list)
    for (fam, arm, s, st), e in ev.items():
        if not (fam == "S" and (arm, s) in bad):
            units[f"{fam}:{st}:{arm}"].append((s, [e]))
    cells = {k: L.Cell.from_evals(k, "gsm_symbolic", k.split(":")[0] + ":" + k.split(":")[2], v) for k, v in units.items()}
    cells["U"] = L.Cell.from_evals("U", "gsm_symbolic", None, [(0, [unt])])
    d = L.Design(cells, strata)
    combos = sorted({tuple(k.split(":")[:2]) for k in cells if k != "U"})

    def fn(v):
        out = {}
        for fam, st in combos:
            ref = f"{fam}:{st}:grpo"
            for k in cells:
                if k.startswith(f"{fam}:{st}:") and k != ref:
                    for o in L.OUTCOMES:
                        out[f"{k}|{o}"] = 100 * (v.m(k, o)[2].mean() - v.m(ref, o)[2].mean())
                    for g in GR:  # accuracy among finished answers only
                        ca = v.m(k, g + "_fin")[2].sum() / max(v.m(k, "stop")[2].sum(), 1e-9)
                        cr = v.m(ref, g + "_fin")[2].sum() / max(v.m(ref, "stop")[2].sum(), 1e-9)
                        out[f"{k}|{g}|finished_only"] = 100 * (ca - cr)
        return out
    obs, reps = L.bootstrap(d, fn, args.B, seed=1508)
    for k in sorted(obs):
        P(f"  {k:40s} {fmt(obs[k], L.ci(reps[k]), 2)} pts")


# ================================================================ prereg 16 (OOD retention)
def load_16(paths):
    data = {}
    for st, files in FILES16.items():
        evs = {}
        for stage, fl in files.items():
            for k, e in L.read_probe([str(paths.p16 / f) for f in fl], DATASET16[st], key_fn=parse_gsm8k_adapter).items():
                evs[(k[1], k[2], k[3])] = e
        data[st] = evs
    return data


def gsm_plus_clusters(ids8, gold8, ids_plus, gold_plus):
    """GSM-Plus testmini (numeric) item -> its seed GSM8K test item by exact question text (HF field 'seed_question').
    Returns (clusters, seed mask over GSM8K items, perturbation type per GSM-Plus item, has-seed mask)."""
    g8 = hf_rows("openai/gsm8k", "main", "test")
    assert all(gsm8k_gold(g8[int(e)]["answer"]) == g for e, g in zip(ids8, gold8)), "GSM8K order mismatch"
    q2pos = {g8[int(e)]["question"]: j for j, e in enumerate(ids8)}
    gp = hf_rows("qintongli/GSM-Plus", None, "testmini", lambda ex: L.numeric_answer(str(ex["answer"])) is not None)
    assert all(L.numeric_answer(str(gp[int(e)]["answer"])) == g for e, g in zip(ids_plus, gold_plus)), "GSM-Plus order mismatch"
    seed_of = np.array([q2pos.get(gp[int(e)]["seed_question"], -1) for e in ids_plus])
    ptype = np.array([gp[int(e)]["perturbation_type"] for e in ids_plus])
    cl = defaultdict(list)
    for j, s in enumerate(seed_of):
        if s >= 0:
            cl[int(s)].append(j)
    clusters = [{"gsm8k": np.array([s]), "gsm_plus": np.array(v)} for s, v in cl.items()]
    seed_mask = np.zeros(len(ids8), bool)
    seed_mask[list(cl)] = True
    return clusters, seed_mask, ptype, seed_of >= 0


def build_16_design(cells, sets_N, clusters, seed_mask, has_seed):
    strata = [("clusters", clusters)] if clusters else []
    strata.append(("items", "gsm8k", np.where(~seed_mask)[0] if clusters else np.arange(sets_N["gsm8k"])))
    if "gsm_plus" in sets_N:
        strata.append(("items", "gsm_plus", np.where(~has_seed)[0] if clusters else np.arange(sets_N["gsm_plus"])))
    for s in ("svamp", "math600"):
        if s in sets_N:
            strata.append(("items", s, np.arange(sets_N[s])))
    return L.Design(cells, [s for s in strata if not (s[0] == "items" and len(s[2]) == 0)])


def stats16_fn(arms_by_source, gate, seed_mask, has_seed, ptypes, tau, graders=GR, with_types=True):
    """Per-set statistics and O1, O2, O4. gate[(set, arm)] = True if the set is scorable for the arm."""
    def q(st, a):
        return (f"{st}:{a}|pre", f"{st}:U", f"{st}:{a}|375", f"{st}:grpo|375")

    def fn(v):
        out = {}
        sets = [s for s in ("gsm8k",) + OOD16 if f"{s}:U" in v.d.cells]
        for o in graders:
            for src, arms in arms_by_source.items():
                for st in sets:
                    for grp, aa in ((f"{src}_pool", arms), *((a, (a,)) for a in ARMS_A if a.startswith(src))):
                        qq = [q(st, a) for a in aa if f"{st}:{a}|pre" in v.d.cells]
                        if not qq:
                            continue
                        for sc, kp in L.SCALES:
                            E = L.eiv_from(L.add_parts([L.eiv_parts(v, o, *x, scale=sc) for x in qq]))
                            W = L.wald_from(L.add_parts([L.wald_parts(v, o, *x, tau, scale=sc) for x in qq]), len(qq))
                            for k in ("b", "beta_S", "beta_U", "VDpre") + (("gap_pre_pts", "gap_post_pts", "rl_alone_pts") if sc == "p" else ()):
                                out[f"{o}|{st}|{grp}|{kp}{k}"] = E[k]
                            for k in ("b_plus", "b_minus"):
                                out[f"{o}|{st}|{grp}|{kp}{k}"] = W[k]
                inc = [(st, a) for st in OOD16 for a in arms if gate.get((st, a)) and f"{st}:{a}|pre" in v.d.cells]
                if inc:
                    for sc, kp in L.SCALES:
                        eo = L.eiv_from(L.add_parts([L.eiv_parts(v, o, *q(st, a), scale=sc) for st, a in inc]))
                        ei = L.eiv_from(L.add_parts([L.eiv_parts(v, o, *q("gsm8k", a), scale=sc) for a in sorted({a for _, a in inc})]))
                        out[f"{o}|O1|{src}|{kp}b_ood"], out[f"{o}|O1|{src}|{kp}b_id"] = eo["b"], ei["b"]
                        out[f"{o}|O1|{src}|{kp}delta"] = eo["b"] - ei["b"]
                gaps = [out.get(f"{o}|{st}|{src}_pool|gap_post_pts", np.nan) for st in OOD16]
                if all(np.isfinite(gaps)) and f"{o}|gsm8k|{src}_pool|gap_post_pts" in out:
                    out[f"{o}|O4|{src}|gap_ood"] = float(np.mean(gaps))
                    out[f"{o}|O4|{src}|delta"] = float(np.mean(gaps)) - out[f"{o}|gsm8k|{src}_pool|gap_post_pts"]
            if "gsm_plus:U" in v.d.cells and seed_mask is not None:
                aa = [a for arms in arms_by_source.values() for a in arms if f"gsm_plus:{a}|pre" in v.d.cells]
                for sc, kp in L.SCALES:
                    ep = L.eiv_from(L.add_parts([L.eiv_parts(v, o, *q("gsm_plus", a), mask=has_seed, scale=sc) for a in aa]))
                    es = L.eiv_from(L.add_parts([L.eiv_parts(v, o, *q("gsm8k", a), mask=seed_mask, scale=sc) for a in aa]))
                    out[f"{o}|O2|{kp}b_pert"], out[f"{o}|O2|{kp}b_seed"] = ep["b"], es["b"]
                    out[f"{o}|O2|{kp}delta"] = ep["b"] - es["b"]
                if with_types and ptypes is not None:
                    for t in sorted(set(ptypes)):
                        e = L.eiv_from(L.add_parts([L.eiv_parts(v, o, *q("gsm_plus", a), mask=ptypes == t) for a in aa]))
                        out[f"{o}|GSMPlus_type|{t}|b"] = e["b"]
        return out
    return fn


def score_16(obs, reps, arms_by_source, label=""):
    res = {}
    for o in GR:
        P(f"\n  [{o}] registered predictions {label} (probability scale; logit in brackets, reported)")
        for src in arms_by_source:
            k = f"{o}|O1|{src}|delta"
            if k not in obs:
                P(f"   O1 {src}: untestable (no OOD set passes the power gate for this source)")
                continue
            lo, hi = L.ci(reps[k])
            held = obs[k] >= REG["O1_MIN"] and lo > 0
            P(f"   O1 {src}: b_OOD {fmt(obs[f'{o}|O1|{src}|b_ood'], L.ci(reps[f'{o}|O1|{src}|b_ood']))} vs b_ID "
              f"{fmt(obs[f'{o}|O1|{src}|b_id'], L.ci(reps[f'{o}|O1|{src}|b_id']))}; delta {fmt(obs[k], (lo, hi))} "
              f"(logit {obs.get(f'{o}|O1|{src}|L_delta', np.nan):+.3f}) -> " + ("HELD" if held else "not held")
              + (" | equivalence: CI inside +-0.10 (H-uniform)" if lo > -REG["HUNIF"] and hi < REG["HUNIF"] else ""))
            res[(o, "O1", src)] = held
            bs = {st: obs.get(f"{o}|{st}|{src}_pool|b", np.nan) for st in OOD16}
            P(f"   O3 {src}: " + " ".join(f"b_{st} {bs[st]:+.3f}" for st in OOD16) + " -> "
              + ("HELD" if bs["math600"] >= bs["svamp"] and bs["math600"] >= bs["gsm_plus"] else "not held"))
            k4 = f"{o}|O4|{src}|delta"
            if src == "teacher" and k4 in obs:
                lo4, hi4 = L.ci(reps[k4])
                P(f"   O4 teacher: OOD gap {obs[f'{o}|O4|{src}|gap_ood']:+.2f} pts; minus GSM8K gap {fmt(obs[k4], (lo4, hi4), 2)} -> "
                  + ("HELD" if obs[k4] >= REG["O4_MIN_PTS"] and lo4 > 0 else "not held"))
        k = f"{o}|O2|delta"
        if k in obs:
            lo, hi = L.ci(reps[k])
            P(f"   O2: b_perturbed {obs[f'{o}|O2|b_pert']:+.3f} vs b_seeds {obs[f'{o}|O2|b_seed']:+.3f}; delta "
              f"{fmt(obs[k], (lo, hi))} -> " + ("HELD" if obs[k] >= REG["O2_MIN"] and lo > 0 else "not held"))
    return res


def part_16(paths, args):
    P("\n# prereg 16 — OOD retention (GSM-Plus, SVAMP, MATH-600) vs GSM8K test (prereg-14 Part A)")
    ev_A, unt_A, e2u = load_A(paths)
    bad_train, _ = part_A_exclusions(paths)
    data = load_16(paths)
    per_set_bad = defaultdict(set)
    all_sets = {"gsm8k": {**ev_A, ("untrained", None, "pre"): unt_A}, **data}
    for st, evs in all_sets.items():
        for (arm, s, stage), e in evs.items():
            if arm != "untrained" and e.a["stop"].mean() < REG["STOP_MIN"]:
                per_set_bad[st].add((arm, s))
    P("eval stop rate < 0.5 (arm, seed) per set:", {k: sorted(v) for k, v in per_set_bad.items()} or "none")
    paired_bad = set().union(*per_set_bad.values()) if per_set_bad else set()
    gp_unt = data["gsm_plus"][("untrained", None, "pre")]
    clusters, seed_mask, ptypes, has_seed = gsm_plus_clusters(unt_A.ids, unt_A.gold, gp_unt.ids, gp_unt.gold)
    P(f"GSM-Plus pairing: {has_seed.sum()}/{len(has_seed)} items matched to {seed_mask.sum()} GSM8K seed questions")
    arms_by_source = {"gold": POOLS["gold_pool"], "teacher": POOLS["teacher_pool"]}
    results = {}
    for mode, label in (("paired", "(primary: an adapter excluded on any set is excluded on every set)"),
                        ("per_set", "(sensitivity: per-set exclusions, the draft's rule)")):
        bad8 = bad_train | (paired_bad if mode == "paired" else per_set_bad["gsm8k"])
        cells = dict(cells_A(ev_A, unt_A, e2u, bad8, steps=(375,), prefix="gsm8k:"))
        sets_N = {"gsm8k": len(unt_A.ids)}
        for st, evs in data.items():
            bad = bad_train | (paired_bad if mode == "paired" else per_set_bad[st])
            cells[f"{st}:U"] = L.Cell.from_evals(f"{st}:U", st, None, [(0, [evs[("untrained", None, "pre")]])])
            for arm in ARMS_A:
                ss = [s for s in SEEDS if (arm, s) not in bad]
                for stage, tag in (("pre", "pre"), (375, "375")):
                    cells[f"{st}:{arm}|{tag}"] = L.Cell.from_evals(f"{st}:{arm}|{tag}", st, arm, [(s, [evs[(arm, s, stage)]]) for s in ss])
            ss = [s for s in SEEDS if ("grpo", s) not in bad]
            cells[f"{st}:grpo|375"] = L.Cell.from_evals(f"{st}:grpo|375", st, "grpo", [(s, [evs[("grpo", s, 375)]]) for s in ss])
            sets_N[st] = len(evs[("untrained", None, "pre")].ids)
        d = build_16_design(cells, sets_N, clusters, seed_mask, has_seed)
        v0 = L.View(d)
        gate = {}
        for st in OOD16:
            for a in ARMS_A:
                if len(cells[f"{st}:{a}|pre"].seeds) >= 2 and len(cells[f"{st}:grpo|375"].seeds) >= 2:
                    vs = L.eiv_from(L.eiv_parts(v0, "len", *quads([a], 375, f"{st}:")[0]))["VDpre"]
                    vi = L.eiv_from(L.eiv_parts(v0, "len", *quads([a], 375, "gsm8k:")[0]))["VDpre"]
                    gate[(st, a)] = bool(vs >= REG["GATE16"] * vi)
                    P(f"  power gate {st} {a}: Var_true(D_pre) {vs:.4f} vs GSM8K {vi:.4f} -> " + ("scorable" if gate[(st, a)] else "UNTESTABLE"))
        obs, reps = L.bootstrap(d, stats16_fn(arms_by_source, gate, seed_mask, has_seed, ptypes, args.tau), args.B, seed=1601)
        for k in sorted(obs):
            if "|O" not in k:
                P(f"  {k:52s} {fmt(obs[k], L.ci(reps[k]))}")
        results[mode] = score_16(obs, reps, arms_by_source, label)
    for st, evs in all_sets.items():  # diagnosis rules
        fin = np.concatenate([e.a["len_fin"].ravel() for e in evs.values()]).astype(bool)
        mk = np.concatenate([e.a["mk_fin"].ravel() for e in evs.values()]).astype(bool)
        P(f"  diagnosis {st}: P(marked-correct | lenient-correct and stopped) {mk[fin].mean():.4f} "
          + ("-> GRADER CHECK (< 0.98)" if mk[fin].mean() < 0.98 else "ok"))
        for (arm, s, stage), e in sorted(evs.items(), key=str):
            if e.a["stop"].mean() < 0.9:
                P(f"    format/truncation flag: {st} {arm} seed {s} {stage} stop rate {e.a['stop'].mean():.3f} (< 0.9)")
    return results


# ================================================================ Part 1 (first-gradient gate) on per-sample phi files
P1_DOC = """Part 1 input (written by the tooling agent) in --p1_dir. Starts are the Part X MERGED-FRESH starts and the
untrained start, all with the same fresh-LoRA initialisation (seed = the E2 run's seed):
  A.jsonl  set A (K=16 per start; T 0.8, top-p 0.95; 600 GSM8K test items fixed by random.Random(15)): one row per
           (item, sample, start, direction_name): item (GSM8K test example_id), sample, start, direction_name, phi,
           correct_lenient, correct_marked [optional 'set': 'A'].
           start in {untrained, gold_mf_s42, gold_mf_s43, gold_mf_s44, teacher_mf_s42, teacher_mf_s43, teacher_mf_s44};
           direction_name in {v0 (raw step-0 policy gradient, merged dW), rand_<j> (random direction in the same LoRA
           geometry, matched on RMS phi), v150 (final merged dW), v25 (merged dW to checkpoint-25)}.
  B.jsonl  set B (K=16, pre-RL success): item, sample, start, correct_lenient, correct_marked.
  C.jsonl  set C (K=16, post-RL success): start in {post_untrained_s42..44 (Part X merged-fresh full_grpo finals),
           post_gold_mf_s42..44, post_teacher_mf_s42..44 (Part X merged-fresh finals)}.
  covariates.jsonl (optional): item, start, name in {silhouette, length, entropy}, value."""


def load_p1(d):
    d = Path(d)
    rows = {}
    for s in "ABC":
        rows[s] = []
        for r in L.read_jsonl(d / f"{s}.jsonl"):
            assert r.get("set", s) == s, f"row of set {r.get('set')} in {s}.jsonl"
            rows[s].append(r)
    covs = list(L.read_jsonl(d / "covariates.jsonl")) if (d / "covariates.jsonl").exists() else []
    return rows, covs


def p1_arrays(rows, covs, g):
    """Per-item arrays for grader g ('correct_lenient' or 'correct_marked')."""
    items = sorted({r["item"] for r in rows["A"]})
    pos = {e: i for i, e in enumerate(items)}
    N = len(items)
    succ = {}
    for s in "ABC":
        acc = defaultdict(dict)
        for r in rows[s]:
            key = (r["item"], r["sample"])
            prev = acc[r["start"]].get(key)
            assert prev is None or prev == bool(r[g]), f"inconsistent correctness {s} {r['start']} {key}"
            acc[r["start"]][key] = bool(r[g])
        for st, dd in acc.items():
            c, n = np.zeros(N), np.zeros(N)
            for (e, _), ok in dd.items():
                if e in pos:
                    c[pos[e]] += ok
                    n[pos[e]] += 1
            succ[(s, st)] = (np.where(n > 0, c / np.maximum(n, 1), np.nan), c, n)
    phi = defaultdict(dict)
    for r in rows["A"]:
        if r.get("direction_name") is not None and r.get("phi") is not None:
            phi[(r["start"], r["direction_name"])][(r["item"], r["sample"])] = (float(r["phi"]), bool(r[g]))
    delta, sbar2 = {}, {}
    for key, dd in phi.items():
        acc1, acc0 = defaultdict(list), defaultdict(list)
        for (e, _), (f, ok) in dd.items():
            (acc1 if ok else acc0)[pos[e]].append(f)
        dl, sb = np.full(N, np.nan), np.full(N, np.nan)
        for i in range(N):
            a1, a0 = acc1.get(i, []), acc0.get(i, [])
            if a1 and a0:
                dl[i] = np.mean(a1) - np.mean(a0)
            if len(a1) >= 2 and len(a0) >= 2:  # class-weighted variance (paper Prop. 5): (1-p) s1^2 + p s0^2
                p = len(a1) / (len(a1) + len(a0))
                sb[i] = (1 - p) * np.var(a1, ddof=1) + p * np.var(a0, ddof=1)
        delta[key], sbar2[key] = dl, sb
    f_unt = np.full(N, np.nan)
    if True:  # untrained unmarked-but-correct rate on set A (a covariate for both graders)
        cl, cm, n, seen = np.zeros(N), np.zeros(N), np.zeros(N), set()
        for r in rows["A"]:
            key = (r["item"], r["sample"])
            if r["start"] == "untrained" and key not in seen:
                seen.add(key)
                cl[pos[r["item"]]] += r["correct_lenient"]
                cm[pos[r["item"]]] += r["correct_marked"]
                n[pos[r["item"]]] += 1
        f_unt = (cl - cm) / np.maximum(n, 1)
    cv = defaultdict(lambda: np.full(N, np.nan))
    for r in covs:
        if r["item"] in pos:
            cv[(r["start"], r["name"])][pos[r["item"]]] = float(r["value"])
    return items, succ, delta, sbar2, phi, f_unt, cv


def part_1(args):
    P("\n# Part 1 — first-gradient mechanism gate (K1', K2', K3) on the Part X merged-fresh starts")
    if not args.p1_dir:
        P(P1_DOC)
        P("no --p1_dir given: Part 1 not scored")
        return
    rows, covs = load_p1(args.p1_dir)
    run_part1(rows, covs, args.B, c_phi=args.c_phi)


def run_part1(rows, covs, B, seed=1510, c_phi=0.5, primary="gold_mf"):
    rng = np.random.default_rng(seed)
    starts = sorted({r["start"] for r in rows["A"] if r["start"] != "untrained"})
    golds = [s for s in starts if s.startswith(primary + "_")]
    for g, gname in (("correct_lenient", "len"), ("correct_marked", "mk")):
        items, succ, delta, sbar2, phi, f_unt, cv = p1_arrays(rows, covs, g)
        N = len(items)
        pA = lambda st: succ[("A", st)][0]
        pB = lambda st: succ[("B", st)][0]
        dirs = sorted({k[1] for k in delta})
        rands = [x for x in dirs if x.startswith("rand")]
        P(f"\n## [{gname}] items {N}; starts {starts} (registered K1'/K2' starts: {golds}); directions {dirs}")
        if gname == "len":
            for st in ["untrained"] + starts:
                for rd in rands:
                    if (st, "v0") in phi and (st, rd) in phi:
                        a = np.sqrt(np.mean([x[0] ** 2 for x in phi[(st, "v0")].values()]))
                        b = np.sqrt(np.mean([x[0] ** 2 for x in phi[(st, rd)].values()]))
                        P(f"  RMS phi {st}: v0 {a:.4g}, {rd} {b:.4g} (ratio {b / a:.3f}; matched if within 10%)")
            # exact crossover condition (paper Prop. 5): SF beats FD iff h sigma_bar < sqrt(2 K_SF / K_FD); v25 is the
            # realised merged step to checkpoint-25 (h = 1); matched compute K_SF (1 + c_phi) = 2 K_FD
            ksf = 16
            kfd = ksf * (1 + c_phi) / 2
            for st in ["untrained"] + starts:
                if (st, "v25") in sbar2:
                    sb = np.sqrt(sbar2[(st, "v25")])
                    ok = np.isfinite(sb)
                    P(f"  crossover {st} (v25, h=1, c_phi {c_phi}): median h*sigma_bar {np.median(sb[ok]):.3f} vs bound "
                      f"{np.sqrt(2 * ksf / kfd):.3f}; share of items where the score-function estimate is more precise "
                      f"{np.mean(sb[ok] < np.sqrt(2 * ksf / kfd)):.2f} ({ok.sum()} items with >= 2 samples per class)")

        def stats(J, wsel):
            out = {}
            k2, k2r = [], []
            for st in starts:
                dsft = (pB(st) - pB("untrained"))[J]
                pS, p0 = pA(st)[J], pA("untrained")[J]
                for dn in dirs:
                    if (st, dn) in delta:
                        out[f"K1|{st}|{dn}"] = L.pspear(delta[(st, dn)][J], dsft, [pS])
                if ("untrained", "v0") in delta:
                    out[f"K1|{st}|placebo_v0_at_theta0"] = L.pspear(delta[("untrained", "v0")][J], dsft, [pS])
                out[f"K1|{st}|rand_mean"] = np.nanmean([out.get(f"K1|{st}|{r}", np.nan) for r in rands]) if rands else np.nan
                post = succ.get(("C", f"post_{st}"))
                if post is not None:
                    y = (post[0] - pB(st))[J]
                    Z = [pS, p0, dsft, f_unt[J]] + [cv[(st, nm)][J] for nm in ("silhouette", "length", "entropy") if (st, nm) in cv]
                    for dn in ["v0"] + rands:
                        if (st, dn) in delta:
                            out[f"K2|{st}|{dn}"] = L.pspear(delta[(st, dn)][J], y, Z)
                    yo = (L.logit_sm(post[1], post[2]) - L.logit_sm(succ[("A", st)][1], succ[("A", st)][2]))[J]
                    out[f"oracle|{st}"] = L.pspear(yo, dsft, [pS])
                    out[f"oracle_share|{st}"] = out.get(f"K1|{st}|v0", np.nan) / out[f"oracle|{st}"]
                    if st in golds:
                        w = wsel[golds.index(st)]
                        k2 += [out.get(f"K2|{st}|v0", np.nan)] * w
                        k2r += [np.nanmean([out.get(f"K2|{st}|{r}", np.nan) for r in rands]) if rands else np.nan] * w
            out["K2|pooled|v0"] = np.nanmean(k2) if k2 else np.nan
            out["K2|pooled|rand"] = np.nanmean(k2r) if k2r else np.nan
            for s in ("42", "43", "44"):
                post = succ.get(("C", f"post_untrained_s{s}"))
                if post is not None and ("untrained", "v0") in delta:
                    out[f"base_forecast|s{s}"] = L.pspear(delta[("untrained", "v0")][J], (post[0] - pB("untrained"))[J], [pA("untrained")[J]])
            return out
        obs = stats(np.arange(N), [1] * len(golds))
        reps = defaultdict(list)
        for _ in range(B):
            J = rng.integers(0, N, N)
            w = np.bincount(rng.integers(0, len(golds), len(golds)), minlength=len(golds)).tolist() if golds else []
            for k, x in stats(J, w).items():
                reps[k].append(x)
        for k in sorted(obs):
            P(f"  {k:44s} {fmt(obs[k], L.ci(reps[k]))}")
        passed = []
        for st in golds:
            s1 = obs.get(f"K1|{st}|v0", np.nan)
            passed.append(bool(s1 <= REG["K1_MAX"] and s1 <= obs.get(f"K1|{st}|rand_mean", np.nan) - REG["K_MARGIN"]
                               and s1 <= obs.get(f"K1|{st}|placebo_v0_at_theta0", np.nan) - REG["K_MARGIN"]))
        P(f"  K1' [{gname}]: {primary} starts passing {sum(passed)}/{len(golds)} -> " + ("HOLDS" if sum(passed) >= 2 else "FAILS (stop the mechanism line)"))
        k2, k2r = obs.get("K2|pooled|v0", np.nan), obs.get("K2|pooled|rand", np.nan)
        P(f"  K2' [{gname}]: pooled {fmt(k2, L.ci(reps['K2|pooled|v0']))} vs random {k2r:+.3f} -> "
          + ("HOLDS" if k2 >= REG["K2_MIN"] and k2 >= k2r + REG["K_MARGIN"] else "FAILS (drop the item-level forecasting claim)"))
        for st in ["untrained"] + starts:
            if (st, "v0") in phi and (st, "v150") in phi:
                ks = sorted(set(phi[(st, "v0")]) & set(phi[(st, "v150")]))
                a = np.array([phi[(st, "v0")][k][0] for k in ks])
                b = np.array([phi[(st, "v150")][k][0] for k in ks])
                obs[f"K3|{st}"] = float(np.corrcoef(a, b)[0, 1])
                P(f"  K3 corr(phi_v0, phi_v150) at {st}: {obs[f'K3|{st}']:+.3f} over {len(ks)} samples")
        k3g = sum(obs.get(f"K3|{st}", -1) >= REG["K3_MIN"] for st in golds)
        P(f"  K3 [{gname}]: untrained {obs.get('K3|untrained', np.nan):+.3f}; {primary} starts >= 0.3: {k3g}/{len(golds)} -> "
          + ("HOLDS" if obs.get("K3|untrained", -1) >= REG["K3_MIN"] and k3g >= 2 else "FAILS (relabel 'early probe'; use v25)"))


# ================================================================ schema-only mode (blinded files)
def schema(paths, part):
    P(f"# schema-only check, part {part}: counts, keys, adapter lists, gold alignment. No outcome field is read.")

    def show(label, paths_):
        info, gold = L.schema_probe(paths_)
        ref = None
        for a, dd in sorted(info.items(), key=lambda x: str(x[0])):
            if ref is None:
                ref = (gold[a], dd["items"])
            ok_gold = all(gold[a][e] == ref[0][e] for e in dd["items"] if e in ref[0])
            P(f"  {label} {a}: rows {dd['rows']}, items {len(dd['items'])}, samples {sorted(dd['samples'])}, "
              f"same item set {dd['items'] == ref[1]}, gold consistent {ok_gold}")
        P(f"  {label} keys: {sorted(set().union(*[d['keys'] for d in info.values()]))}")
        return info, gold
    if part in ("A", "16", "all"):
        allinfo = {}
        for f in ("sft", "s75", "s225", "s375", "untrained"):
            info, gold = show(f"14a/{f}", str(paths.p14a / f"{f}.jsonl"))
            allinfo.update(info)
            if f == "untrained":
                g0 = gold[None]
        keys = [parse_gsm8k_adapter(a) for a in allinfo]
        P(f"  parsed Part A adapters: {len(keys)}; cells: {sorted({(k[0], k[1], k[3]) for k in keys}, key=str)}")
        for s in SEEDS:
            rows, gold, kk = L.schema_job(f"{paths.e2}grpo/seed_{s}/base_predictions.jsonl")
            P(f"  E2 base seed {s}: rows {rows}, gold mismatches vs probe example_ids {sum(gold[e] != g0[e] for e in g0)}")
        miss = [(a, s) for a in ARMS_A + ("grpo",) for s in SEEDS
                if not (paths.sprime["teacher" if a.startswith("teacher") else "gold"]
                        / f"seed_{s}/{'full_grpo' if a == 'grpo' else 'random_sft_' + a.split('_')[1]}/grpo/checkpoint-375/trainer_state.json").exists()]
        P(f"  trainer_state.json missing for: {miss or 'none'}")
    if part in ("0", "all"):
        info, gold = show("14c/ckpt", str(paths.p14c / "ckpt.jsonl"))
        P(f"  parsed: {sorted({(parse_math_adapter(a)[0], parse_math_adapter(a)[2]) for a in info}, key=str)}")
        g0 = next(iter(gold.values()))
        for s in SEEDS:
            for f in ("base_predictions", "full_grpo/predictions", "random_sft_250/predictions", "random_sft_250/predictions_after_sft"):
                rows, gold_j, kk = L.schema_job(str(paths.math / f"seed_{s}/{f}.jsonl"))
                P(f"  MATH seed {s} {f}: rows {rows}, gold mismatches vs ckpt example_ids {sum(gold_j[e] != g0[e] for e in g0)}")
            for meth in ("full_grpo", "random_sft_250"):
                P(f"  rollouts {s}/{meth}: exists {(paths.math / f'seed_{s}/{meth}/grpo/rollouts.jsonl').exists()}")
    if part in ("B", "all"):
        for f in ("c1ep", "s225", "s375", "untrained"):
            info, gold = show(f"14b/{f}", str(paths.p14b / f"{f}.jsonl"))
        try:
            ds = hf_rows("apple/GSM-Symbolic", "main", "test")
            ok = all(gsm8k_gold(ds[int(e)]["answer"]) == g for e, g in gold[None].items())
            P(f"  GSM-Symbolic HF order check (gold of shuffled(1234) rows == probe gold): {ok}; templates "
              f"{len(set(ds['original_id']))}")
        except Exception as e:
            P("  GSM-Symbolic HF check failed:", repr(e))
    if part in ("X", "all"):
        for (src, proto), p in paths.px.items():
            for s in SEEDS:
                fs = [p / f"seed_{s}/random_sft_250/{f}" for f in ("predictions_after_sft.jsonl", "predictions.jsonl", "grpo/rollouts.jsonl")]
                P(f"  Part X {src}/{proto} seed {s}: " + ", ".join(f"{f.name} {'ok' if f.exists() else 'MISSING'}" for f in fs))
        for s in SEEDS:
            fs = [paths.px[("gold", "mf")] / f"seed_{s}/full_grpo/{f}" for f in ("predictions.jsonl", "grpo/rollouts.jsonl", "grpo/merged_start.json")]
            P(f"  Part X RL alone (merged-fresh full_grpo) seed {s}: " + ", ".join(f"{f.name} {'ok' if f.exists() else 'MISSING'}" for f in fs))
    if part in ("16", "all"):
        for st, files in FILES16.items():
            for stage, fl in files.items():
                ex = [paths.p16 / f for f in fl]
                if all(p.exists() for p in ex):
                    show(f"16/{st}/{stage}", [str(p) for p in ex])
                else:
                    P(f"  16/{st}/{stage}: missing {[p.name for p in ex if not p.exists()]} (prereg 16 not yet run)")
        try:
            g8 = hf_rows("openai/gsm8k", "main", "test")
            gp = hf_rows("qintongli/GSM-Plus", None, "testmini", lambda ex: L.numeric_answer(str(ex["answer"])) is not None)
            q8 = set(g8["question"])
            P(f"  GSM-Plus testmini numeric items {len(gp)}; seed_question exact matches in GSM8K test "
              f"{sum(q in q8 for q in gp['seed_question'])}; distinct seeds {len(set(gp['seed_question']))}; "
              f"perturbation types {sorted(set(gp['perturbation_type']))}")
        except Exception as e:
            P("  GSM-Plus HF check failed:", repr(e))


# ================================================================ dry run: E2 reproduction
def e2_design(paths):
    U = [L.read_job(f"{paths.e2}grpo/seed_{s}/base_predictions.jsonl", "gsm8k") for s in SEEDS]
    G0 = {s: L.read_job(f"{paths.e2}grpo/seed_{s}/random_sft_250/predictions_after_sft.jsonl", "gsm8k") for s in SEEDS}
    U1 = {s: [L.read_job(f"{paths.e2}{lt}/seed_{s}/full_grpo/predictions.jsonl", "gsm8k") for lt in ("grpo", "dr_grpo")] for s in SEEDS}
    G1 = {s: [L.read_job(f"{paths.e2}{lt}/seed_{s}/random_sft_250/predictions.jsonl", "gsm8k") for lt in ("grpo", "dr_grpo")] for s in SEEDS}
    cells = {"U": L.Cell.from_evals("U", "gsm8k", None, [(i, [e]) for i, e in enumerate(U)]),
             "gold_250|pre": L.Cell.from_evals("gold_250|pre", "gsm8k", "gold_250", [(s, [G0[s]]) for s in SEEDS]),
             "gold_250|150": L.Cell.from_evals("gold_250|150", "gsm8k", "gold_250", [(s, G1[s]) for s in SEEDS]),
             "grpo|150": L.Cell.from_evals("grpo|150", "gsm8k", "grpo", [(s, U1[s]) for s in SEEDS])}
    return L.Design(cells)


def dry_e2(paths, args):
    P("# DRY RUN: prereg-13 E2 (already analysed; unblinded). Reproduce G1/R2 slopes and R1 bins / pool gap.")
    t0 = time.time()
    d = e2_design(paths)
    P(f"marked regrade vs stored 'correct': {L.AGREE[True]}/{sum(L.AGREE.values())} = "
      f"{L.AGREE[True] / sum(L.AGREE.values()):.5f} (must be 1)")
    try:
        sys.path.insert(0, str(paths.root))
        import route_sft_grpo as R
        n = bad = 0
        for s in SEEDS:
            for f in (f"{paths.e2}grpo/seed_{s}/base_predictions.jsonl", f"{paths.e2}grpo/seed_{s}/random_sft_250/predictions.jsonl"):
                for r in L.read_jsonl(f):
                    for mk in (False, True):
                        n += 1
                        bad += R.answer_is_correct(r["completion"], r["gold"], mk, "gsm8k") != L.answer_is_correct(r["completion"], r["gold"], mk, "gsm8k")
        P(f"port check vs route_sft_grpo.answer_is_correct (lenient and marked): {n - bad}/{n} identical")
    except Exception as e:
        P("port check vs route_sft_grpo skipped:", repr(e))
    q = ("gold_250|pre", "U", "gold_250|150", "grpo|150")
    v = L.View(d)
    for o, target in (("mk", 0.4566), ("len", 0.5623)):
        E = L.eiv_from(L.eiv_parts(v, o, *q))
        P(f"EIV retention b [{o}] = {E['b']:.4f} (target {target}; |diff| {abs(E['b'] - target):.4f} -> "
          + ("OK" if abs(E["b"] - target) <= 0.01 else "FAIL") + f"); 1 + beta_S - beta_U = {1 + E['beta_S'] - E['beta_U']:.4f}")
    bn = L.bins_from(L.bin_parts(v, "len", *q))
    P(f"R1 bins [len]: 6/6 bin n {bn['bin_all_n']:.0f} D0 {100 * bn['bin_all_D0']:+.2f} D150 {100 * bn['bin_all_D1']:+.2f} "
      f"retention {bn['bin_all_ret']:.3f} (target 0.42 -> " + ("OK" if abs(bn["bin_all_ret"] - 0.42) < 0.006 else "FAIL") + ")")
    pairs = [(f"{paths.e2}{lt}/seed_{s}/random_sft_250/grpo/rollouts.jsonl", f"{paths.e2}{lt}/seed_{s}/full_grpo/grpo/rollouts.jsonl")
             for lt in ("grpo", "dr_grpo") for s in SEEDS]
    g0, g1 = pool_gap(pairs, (0, 25), (125, 150))
    P(f"R1 pool gap G - U: first block {g0.mean():+.3f} (target -0.090), last block {g1.mean():+.3f} (target -0.100) -> "
      + ("OK" if abs(g0.mean() + 0.090) < 0.0015 and abs(g1.mean() + 0.100) < 0.0015 else "FAIL") + f"; ratio {g1.mean() / g0.mean():+.2f}")
    P("E2 stop rate, last 20 steps:", {f"{lt}/{s}/{m}": round(stop_rate_rollouts(f"{paths.e2}{lt}/seed_{s}/{m}/grpo/rollouts.jsonl")[0], 3)
                                       for lt in ("grpo", "dr_grpo") for s in SEEDS for m in ("full_grpo", "random_sft_250")})
    P(f"(load + checks {time.time() - t0:.0f}s)\n\n## full Part-A machinery on E2 (B={args.B}, nsim={args.nsim}, tau={args.tau:.3f})")
    G = {"gold_250": ("gold_250",)}
    t0 = time.time()
    obs, reps = L.bootstrap(d, stats_fn(G, (150,), GR, args.tau), args.B, seed=1513)
    P(f"(bootstrap {time.time() - t0:.0f}s)")
    for o in GR:
        t0 = time.time()
        nl, _ = run_nulls(d, o, {"gold_250": q}, G, args.nsim, args.tau, seed=1514)
        P(f"  [{o}] E2 step 150 gold_250 (nulls {time.time() - t0:.0f}s)")
        nn = {k: v for (k, g), v in nl.items()}
        print_group(obs, reps, f"{o}|150|gold_250|", nn)
        val, cint, lab = bin_excess(obs, reps, nn, f"{o}|150|gold_250|")
        P(f"    base-success contrast as excess over the per-bin one-index null (bins {lab}): {fmt(val, cint)}")
        vl, cl, _ = bin_excess(obs, reps, nn, f"{o}|150|gold_250|", kp="L_")
        P(f"    same on the logit scale: {fmt(vl, cl)}")
        m = med(nn["memsym"]["M"])
        lo, hi = L.ci(np.asarray(reps[f"{o}|150|gold_250|M"]) - m)
        P(f"    memory: M - symmetric-null median = {obs[f'{o}|150|gold_250|M'] - m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


# ================================================================ dry runs: synthetic worlds
def _binom_units(rng, name, lineage, probs, k, set_name="gsm8k", seeds=SEEDS):
    U, N = len(probs), len(probs[0])
    C = np.zeros((2, U, N))
    for u, p in enumerate(probs):
        C[0, u] = rng.binomial(k // 2, p)
        C[1, u] = rng.binomial(k - k // 2, p)
    Nh = np.array([[k // 2] * U, [k - k // 2] * U], float)
    return L.Cell(name, set_name, lineage, list(seeds)[:U] if U <= len(seeds) else list(range(U)), {"len": C}, Nh)


def synth_A(rng, scen, N=1319, nU=4, k=4, arms=("gold_250", "gold_500", "teacher_250", "teacher_500"), b0=0.5, r0=0.4,
            off=0.0, sig=1.3, kappa=0.5, mem=0.6, seed_sd=0.15):
    """Part-A-sized synthetic data with a planted truth (default noise = E2-like: Var_true(D_pre) ~ 0.04-0.06).
    Probability-scale worlds: 'sym' (q_S - q_R = b0 D_pre + off), 'repair' (D_pre for D_pre > 0, r0 D_pre for
    D_pre < 0), 'partial' (0.8 helped, 0.5 hurt). Logit worlds with a shared item trait th: 'tilt' (one index,
    q = expit(logit p + 1) for both starts), 'trait_sym' (p_U = th + e_U, p_S = th + mu + e_S, independent
    deviations, each start keeps kappa of its own deviation: no history asymmetry), 'attract_base' (p_S = p_U +
    delta; RL pulls both starts to th: SFT effects shrink toward the base profile, M > 0), 'attract_sft' (RL alone
    acquires the teacher SFT effects: M < 0), 'trait_mem' (trait_sym + mem * e_U in q_S: M > 0)."""
    th = rng.normal(-0.15, 2.3, N)
    eU = rng.normal(0, 0.8, N)
    lpU = th + eU
    T = {"U": L.expit(lpU)}
    dl = {"gold": rng.normal(-0.7, sig, N), "teacher": rng.normal(0.5, sig, N)}
    cells = {"U": _binom_units(rng, "U", None, [T["U"]] * nU, k, seeds=range(nU))}
    cells["U"].lineage = None
    qR_s = {}
    for s in SEEDS:
        es = rng.normal(0, seed_sd, N)
        if scen in ("trait_sym", "trait_mem", "attract_base"):
            qR_s[s] = L.expit(th + 1.0 + kappa * eU + es + rng.normal(0, 0.3, N))
        elif scen == "attract_sft":
            qR_s[s] = L.expit(th + 1.0 + kappa * eU + dl["teacher"] + es + rng.normal(0, 0.3, N))
        elif scen == "tilt":
            qR_s[s] = L.expit(lpU + 1.0 + es)
        else:
            qR_s[s] = L.expit(th + 1.0 + rng.normal(0, 0.5, N) + es)
    for arm in arms:
        src, n = arm.split("_")
        delta = dl[src] + (rng.normal(0, 0.5, N) if n == "500" else 0)
        mu = 0.4 if src == "teacher" else -0.4
        eS = rng.normal(0, 0.8, N)
        pss, qss = [], []
        for s in SEEDS:
            lp = (th + mu + eS if scen in ("trait_sym", "trait_mem") else lpU + delta) + rng.normal(0, seed_sd, N)
            ps = L.expit(lp)
            D = ps - T["U"]
            if scen == "sym":
                qs = qR_s[s] + b0 * D + off
            elif scen == "repair":
                qs = qR_s[s] + np.where(D > 0, D, r0 * D)
            elif scen == "partial":
                qs = qR_s[s] + np.where(D > 0, 0.8 * D, 0.5 * D)
            elif scen == "tilt":
                qs = L.expit(lp + 1.0)
            elif scen == "trait_sym":
                qs = L.expit(th + 1.0 + kappa * (mu + eS) + rng.normal(0, 0.3, N))
            elif scen == "trait_mem":
                qs = L.expit(th + 1.0 + kappa * (mu + eS) + mem * eU + rng.normal(0, 0.3, N))
            elif scen == "attract_base":
                qs = L.expit(th + 1.0 + kappa * (eU + delta) + rng.normal(0, 0.3, N))
            elif scen == "attract_sft":
                qs = L.expit(th + 1.0 + kappa * eU + delta + rng.normal(0, 0.3, N))
            elif scen == "lsym":  # symmetric on the logit scale: each start keeps kappa of its logit effect
                lq = np.log(qR_s[s]) - np.log1p(-qR_s[s])
                qs = L.expit(lq + kappa * (lp - lpU))
            else:
                raise ValueError(scen)
            pss.append(ps)
            qss.append(np.clip(qs, 0.0, 1.0))
        cells[f"{arm}|pre"] = _binom_units(rng, f"{arm}|pre", arm, pss, k)
        cells[f"{arm}|375"] = _binom_units(rng, f"{arm}|375", arm, qss, k)
        T[f"{arm}|pre"], T[f"{arm}|375"] = np.mean(pss, 0), np.mean(qss, 0)
    qr = [qR_s[s] for s in SEEDS]
    cells["grpo|375"] = _binom_units(rng, "grpo|375", "grpo", qr, k)
    T["grpo|375"] = np.mean(qr, 0)
    return L.Design(cells), T


def truth_stats(v, T, q, tau):
    """Estimands: the same statistics on the TRUE per-item probabilities (probability scale); Wald and bins keep the
    noisy half-sample selection, since their estimand is conditional on it (paper Prop. 7d)."""
    S, U, PS, PR = q
    pS, pU, qS, qR = T[S], T[U], T[PS], T[PR]
    Dpre, Dpost = pS - pU, qS - qR
    V = L.cov(Dpre, Dpre)
    out = {"b": L.cov(Dpost, Dpre) / V, "beta_S": L.cov(qS - pS, Dpre) / V, "beta_U": L.cov(qR - pU, Dpre) / V}
    s0, s1, _, u0, u1, _, _, _ = L.pair_means(v, "len", *q)
    acc = {"p": [0.0, 0.0], "m": [0.0, 0.0], "po": [0.0, 0.0], "mo": [0.0, 0.0]}
    for f in (0, 1):
        sel = (s0 - u0) if f == 0 else (s1 - u1)
        plus, minus = sel >= tau - 1e-9, sel <= -tau + 1e-9
        zero = ~(plus | minus)
        for sd, mk in (("p", plus), ("m", minus)):
            if mk.sum() and zero.sum():
                acc[sd][0] += Dpost[mk].mean() - Dpost[zero].mean()
                acc[sd][1] += Dpre[mk].mean() - Dpre[zero].mean()
                acc[sd + "o"][0] += Dpost[mk].mean()
                acc[sd + "o"][1] += Dpre[mk].mean()
    out["b_plus"], out["b_minus"] = acc["p"][0] / acc["p"][1], acc["m"][0] / acc["m"][1]
    out["plus_num"], out["plus_den"], out["minus_num"], out["minus_den"] = acc["p"][0], acc["p"][1], acc["m"][0], acc["m"][1]
    out["origin_plus"], out["origin_minus"] = acc["po"][0] / acc["po"][1], acc["mo"][0] / acc["mo"][1]
    X = np.column_stack([np.ones(len(pS)), pS, pU])
    out["lam_S"] = np.linalg.lstsq(X, qS, rcond=None)[0][2]
    out["lam_R"] = np.linalg.lstsq(np.column_stack([np.ones(len(pS)), pU, pS]), qR, rcond=None)[0][2]
    out["M"] = out["lam_S"] - out["lam_R"]
    lab = L.bin_index(u0)
    out["bin_all_ret"] = Dpost[lab == 4].mean() / Dpre[lab == 4].mean() if (lab == 4).sum() else np.nan
    return out


def naive_stats(v, q, tau):
    """Without the safeguards: OLS on the noisy full-sample D_pre (attenuation), selection and measurement on the
    same samples (regression to the mean), OLS memory regression on noisy p's, beta's with shared samples."""
    s0, s1, s, u0, u1, u, qs, qr = L.pair_means(v, "len", *q)
    D, post = s - u, qs - qr
    out = {"b": L.cov(post, D) / L.cov(D, D), "beta_S": L.cov(qs - s, D) / L.cov(D, D), "beta_U": L.cov(qr - u, D) / L.cov(D, D)}
    plus, minus = D >= TAU - 1e-9, D <= -TAU + 1e-9
    zero = ~(plus | minus)
    out["b_plus"] = (post[plus].mean() - post[zero].mean()) / (D[plus].mean() - D[zero].mean())
    out["b_minus"] = (post[minus].mean() - post[zero].mean()) / (D[minus].mean() - D[zero].mean())
    out["lam_S"] = np.linalg.lstsq(np.column_stack([np.ones(len(s)), s, u]), qs, rcond=None)[0][2]
    out["lam_R"] = np.linalg.lstsq(np.column_stack([np.ones(len(s)), u, s]), qr, rcond=None)[0][2]
    out["M"] = out["lam_S"] - out["lam_R"]
    return out


def classify_rates(bp, bm, sd_, sb, rb):
    d = bp - bm
    hs = (bp <= sb) & (np.abs(d) < sd_)
    hr = (bp >= rb) & (d >= sd_)
    return hs.mean(), hr.mean()


SYM_WORLDS = [("sym", {"b0": 0.4}), ("sym", {"b0": 0.5}), ("sym", {"b0": 0.6}), ("sym", {"b0": 0.5, "off": -0.03}),
              ("sym", {"b0": 0.5, "off": 0.03})]
REP_WORLDS = [("repair", {"r0": 0.3}), ("repair", {"r0": 0.45}), ("repair", {"r0": 0.6})]


def dry_power(args):
    """Calibration of the sign split (lead's change 3): b+ and b- on noisy halves are pulled toward each other, so the
    H-sym / H-repair thresholds are set from symmetric and repair-only worlds passed through the identical pipeline."""
    P("# DRY RUN (power): sign-split calibration. Part-A sizes: 1319 items; untrained 16 samples (4 if 'U 4');"
      " each arm 3 seeds x 4 samples pre and post; GRPO alone 3 x 4; pooled n=250 + n=500 per source")
    rng = np.random.default_rng(1516)
    taus = (0.2, 1 / 3, 0.5)
    sims = {}
    for nU in (4, 1):
        for sig in (1.3, 0.8):
            for scen, kw in SYM_WORLDS + REP_WORLDS + [("partial", {})]:
                rr = defaultdict(list)
                for _ in range(args.power_reps):
                    d, T = synth_A(rng, scen, nU=nU, sig=sig, **kw)
                    v = L.View(d)
                    for t in taus:
                        WT = L.wald_from(L.add_parts([L.wald_parts(v, "len", *q, t) for q in quads(POOLS["teacher_pool"], 375)]), 2)
                        WG = L.wald_from(L.add_parts([L.wald_parts(v, "len", *q, t) for q in quads(POOLS["gold_pool"], 375)]), 2)
                        rr[t].append((WT["b_plus"], WG["b_minus"], WT["origin_plus"], WG["origin_minus"], WT["n_plus"], WG["n_minus"]))
                    if len(rr[taus[0]]) == 1:
                        E = L.eiv_from(L.add_parts([L.eiv_parts(v, "len", *q) for q in quads(POOLS["teacher_pool"], 375)]))
                        EG = L.eiv_from(L.add_parts([L.eiv_parts(v, "len", *q) for q in quads(POOLS["gold_pool"], 375)]))
                        P(f"  [world {scen} {kw} sigma {sig} U {4 * nU}] first draw: Var_true(D_pre) teacher {E['VDpre']:.4f} "
                          f"gold {EG['VDpre']:.4f}; mean D_pre teacher {E['gap_pre_pts']:+.1f} gold {EG['gap_pre_pts']:+.1f} pts; "
                          f"b teacher {E['b']:+.3f} gold {EG['b']:+.3f}")
                for t in taus:
                    sims[(nU, sig, scen, str(kw), t)] = np.array(rr[t], float)
    for est, (ip, im) in (("Wald chord (primary)", (0, 1)), ("through origin (Def. 1)", (2, 3))):
        P(f"\n## estimator: {est}")
        for (nU, sig, scen, kw, t), a in sims.items():
            bp, bm = a[:, ip], a[:, im]
            sd = np.nanstd(bp - bm)
            P(f"  U {4 * nU:2d} sigma {sig} tau {t:.2f} {scen:7s} {kw:26s}: b+(T) {np.nanmean(bp):+.3f} (SD {np.nanstd(bp):.3f}) "
              f"b-(G) {np.nanmean(bm):+.3f} (SD {np.nanstd(bm):.3f}) d {np.nanmean(bp - bm):+.3f} (SD {sd:.3f}; MDE80 {2.8 * sd:.2f}) "
              f"items +/- per arm {np.nanmean(a[:, 4]):.0f}/{np.nanmean(a[:, 5]):.0f}")
    P("\n## threshold search (Wald chord, U 16): rule H-sym: b+(T) <= SYM_BP and |d| < SPLIT_D; H-repair: b+(T) >= REP_BP"
      " and d >= SPLIT_D. Objective: min over sym worlds of P(H-sym) and over repair worlds of P(H-repair).")
    best = {}
    for sig in (1.3, 0.8):
        for t in taus:
            res = []
            for sd_ in np.arange(0.05, 0.41, 0.025):
                for sb in np.arange(0.6, 0.96, 0.05):
                    for rb in np.arange(0.5, 0.96, 0.05):
                        ps = [classify_rates(sims[(4, sig, s, str(k), t)][:, 0], sims[(4, sig, s, str(k), t)][:, 1], sd_, sb, rb)[0] for s, k in SYM_WORLDS]
                        pr = [classify_rates(sims[(4, sig, s, str(k), t)][:, 0], sims[(4, sig, s, str(k), t)][:, 1], sd_, sb, rb)[1] for s, k in REP_WORLDS]
                        res.append((min(ps + pr), sd_, sb, rb, ps, pr))
            res.sort(key=lambda x: -x[0])
            best[(sig, t)] = res[0]
            P(f"  sigma {sig} tau {t:.2f}: best min-P {res[0][0]:.2f} at SPLIT_D {res[0][1]:.3f} SYM_BP {res[0][2]:.2f} "
              f"REP_BP {res[0][3]:.2f}; P(H-sym) per sym world {np.round(res[0][4], 2).tolist()}; P(H-repair) per repair "
              f"world {np.round(res[0][5], 2).tolist()}")
    P("\n## registered thresholds (REG) evaluated in every world:")
    for (nU, sig, scen, kw, t), a in sims.items():
        if abs(t - args.tau) < 1e-9:
            hs, hr = classify_rates(a[:, 0], a[:, 1], REG["SPLIT_D"], REG["SYM_BP"], REG["REP_BP"])
            P(f"  U {4 * nU:2d} sigma {sig} {scen:7s} {kw:26s}: P(H-sym) {hs:.2f} P(H-repair) {hr:.2f} P(intermediate) {1 - hs - hr:.2f}")


PC_WORLDS = [("sym", {"b0": 0.5}, "H-sym"), ("sym", {"b0": 0.6}, "H-sym"), ("sym", {"b0": 0.5, "off": -0.03}, "H-sym"),
             ("lsym", {"kappa": 0.5}, "H-sym"), ("repair", {"r0": 0.45}, "H-repair"), ("repair", {"r0": 0.6}, "H-repair"),
             ("partial", {}, "-")]


def _pc_worker(job):
    """One synthetic Part-A dataset: fixed-threshold verdict and the verdict after the registered location check."""
    scen, kw, seed = job
    rng = np.random.default_rng(seed)
    d, T = synth_A(rng, scen, **kw)
    v = L.View(d)
    obs = {}
    for g, arms in POOLS.items():
        obs.update(L.all_stats(v, "len", quads(arms, 375), TAU, prefix=f"len|375|{g}|", scales=(("p", ""),)))
    bp, bm = obs["len|375|teacher_pool|b_plus"], obs["len|375|gold_pool|b_minus"]
    v0 = classify_split(bp, bm)
    v1 = v0
    if v0.startswith("H-"):
        aq = {a: quads([a], 375)[0] for a in POOLS["teacher_pool"] + POOLS["gold_pool"]}
        models = {}
        for a, (S, U, PS, PR) in aq.items():
            cU, nU = L.counts(d.cells[U], "len")
            cS, nS = L.counts(d.cells[S], "len")
            cr, nr = L.counts(d.cells[PR], "len")
            lat = L.Latent2D(cU, nU, cS, nS)
            models[a] = {"lat": lat, "sU": L.fit_tilt(lat, lat.A, cr, nr)}
        v1 = split_location_check(d, "len", models, aq, obs, 375, "teacher_pool", "gold_pool", POOLS["teacher_pool"],
                                  POOLS["gold_pool"], v0, 200, TAU, seed=seed + 1, log=False)
    return scen, str(kw), v0, v1, bp - bm


def dry_power_check(args):
    """Power cost of the registered location check (lead's request): correct-classification rates of the fixed rule
    alone and of the fixed rule followed by the location check (registered settings: 200 replicates per world, matching
    by bisection), at Part-A sizes with E2-sized effects; 'lsym' is symmetric on the logit scale."""
    from multiprocessing import Pool
    P(f"# DRY RUN (power_check): {args.power_reps} datasets per world, {args.procs} processes")
    jobs = [(s, kw, 15170 + 1000 * i + r) for i, (s, kw, _) in enumerate(PC_WORLDS) for r in range(args.power_reps)]
    with Pool(args.procs) as pool:
        res = pool.map(_pc_worker, jobs, chunksize=1)
    for s, kw, truth in PC_WORLDS:
        rr = [x for x in res if x[0] == s and x[1] == str(kw)]
        for lab, j in (("fixed rule", 2), ("fixed rule + location check", 3)):
            ps = np.mean([x[j] == "H-sym" for x in rr])
            pr = np.mean([x[j].startswith("H-repair") for x in rr])
            P(f"  {s:7s} {str(kw):26s} (truth {truth:8s}) {lab:28s}: P(H-sym) {ps:.2f} P(H-repair) {pr:.2f} "
              f"P(intermediate) {1 - ps - pr:.2f}" + (f"  -> P(correct) {ps if truth == 'H-sym' else pr:.2f}" if truth != "-" else ""))
        P(f"  {'':7s} {'':26s} mean d {np.mean([x[4] for x in rr]):+.3f} (SD {np.std([x[4] for x in rr]):.3f}), n = {len(rr)}")


def dry_synthetic(args):
    P("# DRY RUN (synthetic): planted truths through the identical pipeline (Part-A sizes, see dry_run power)")
    rng = np.random.default_rng(1515)
    tau = args.tau
    qT = ("teacher_250|pre", "U", "teacher_250|375", "grpo|375")
    qG = ("gold_250|pre", "U", "gold_250|375", "grpo|375")
    if "1" in args.sections:
        rng = np.random.default_rng(1515 + 1)  # independent stream per section (sections may run in separate jobs)
        P(f"\n## 1. recovery: mean(estimate - estimand) over {args.reps} datasets (MC SE), single arm n=250; naive = without safeguards")
        keys = ("b", "beta_S", "beta_U", "b_plus", "b_minus", "origin_plus", "origin_minus", "lam_S", "lam_R", "M", "bin_all_ret")
        for scen in ("sym", "repair", "partial", "tilt", "trait_sym", "attract_base", "attract_sft", "trait_mem"):
            acc = defaultdict(list)
            for _ in range(args.reps):
                d, T = synth_A(rng, scen)
                v = L.View(d)
                for lab, q in (("T", qT), ("G", qG)):
                    est = L.all_stats(v, "len", [q], tau, scales=(("p", ""),))
                    tru, nai = truth_stats(v, T, q, tau), naive_stats(v, q, tau)
                    for k in keys:
                        acc[(lab, k, "e")].append(est[k])
                        acc[(lab, k, "t")].append(tru[k])
                        acc[(lab, k, "n")].append(nai.get(k, np.nan))
            P(f"\n  world '{scen}':")
            for lab in ("T", "G"):
                for k in keys:
                    e, t, n = (np.array(acc[(lab, k, x)], float) for x in "etn")
                    ok = np.isfinite(e) & np.isfinite(t)
                    if ok.sum() < 5:
                        continue
                    dd = e[ok] - t[ok]
                    line = (f"    {'teacher' if lab == 'T' else 'gold   '}_250 {k:13s} estimand {t[ok].mean():+.3f} est {e[ok].mean():+.3f} "
                            f"bias {dd.mean():+.4f} ({dd.std() / np.sqrt(ok.sum()):.4f}) SD {e[ok].std():.3f}")
                    if np.isfinite(n[ok]).any():
                        line += f" | naive bias {np.nanmean(n[ok] - t[ok]):+.3f}"
                    P(line)
    if "2" in args.sections:
        rng = np.random.default_rng(1515 + 2)  # independent stream per section (sections may run in separate jobs)
        P("\n## 2. null calibration: observed statistic vs its simulated null fitted on the same synthetic data")
        for scen, kind, stat in (("tilt", "tilt", "b"), ("tilt", "tilt", "b_plus"), ("tilt", "tilt", "bin_all_ret"),
                                 ("tilt", "tilt", "L_b"), ("trait_sym", "memsym", "M"), ("attract_base", "memsym", "M"),
                                 ("trait_mem", "memsym", "M"), ("attract_sft", "memsym", "M")):
            zs = []
            for _ in range(args.null_reps):
                d, T = synth_A(rng, scen)
                nl, _ = run_nulls(d, "len", {"x": qT}, {"x": ("x",)}, args.nsim, tau, seed=int(rng.integers(1 << 30)), kinds=(kind,), log=False)
                obs = L.all_stats(L.View(d), "len", [qT], tau)
                nd = nl[(kind, "x")][stat]
                nd = nd[np.isfinite(nd)]
                zs.append(((obs[stat] - np.median(nd)) / nd.std(), np.mean(nd >= obs[stat])))
            z = np.array(zs)
            P(f"  world '{scen}', {kind} null, {stat}: (obs - null median)/null SD {np.round(z[:, 0], 2).tolist()}; "
              f"P(null >= obs) {np.round(z[:, 1], 3).tolist()}" + ("  [null true: centred]" if scen in ("tilt", "trait_sym") else "  [null false]"))
    if "3" in args.sections:
        rng = np.random.default_rng(1515 + 3)  # independent stream per section (sections may run in separate jobs)
        P(f"\n## 3. coverage of the 95% hierarchical bootstrap CI ('sym' world, {args.cov_reps} datasets, B={args.cov_B})")
        hit = defaultdict(list)
        for _ in range(args.cov_reps):
            d, T = synth_A(rng, "sym")
            base = stats_fn(POOLS, (375,), ("len",), tau)

            def fn(v):  # registered statistics, plus the sign split on the item-only twin (comparison)
                out = base(v)
                for g, arms in POOLS.items():
                    W = L.wald_from(L.add_parts([L.wald_parts(v.vi, "len", *q, tau) for q in quads(arms, 375)]), 2)
                    out[f"len|375|{g}|b_plus_item"], out[f"len|375|{g}|b_minus_item"] = W["b_plus"], W["b_minus"]
                return out
            obs, reps = L.bootstrap(d, fn, args.cov_B, seed=int(rng.integers(1 << 30)))
            v = L.View(d)
            for g, arms in POOLS.items():
                qq = quads(arms, 375)
                num = sum(L.cov(T[q[2]] - T[q[3]], T[q[0]] - T[q[1]]) for q in qq)
                den = sum(L.cov(T[q[0]] - T[q[1]], T[q[0]] - T[q[1]]) for q in qq)
                tr = [truth_stats(v, T, q, tau) for q in qq]
                # pooled Wald estimand = ratio of the summed true chords (as the estimator pools)
                tp = {side: sum(x[f"{side}_num"] for x in tr) / sum(x[f"{side}_den"] for x in tr) for side in ("plus", "minus")}
                for k, t in (("b", num / den), ("b_plus", tp["plus"]), ("b_minus", tp["minus"]),
                             ("b_plus_item", tp["plus"]), ("b_minus_item", tp["minus"])):
                    lo, hi = L.ci(reps[f"len|375|{g}|{k}"])
                    hit[(g, k)].append(lo <= t <= hi)
        for (g, k), h in sorted(hit.items()):
            P(f"  {g} {k}: coverage {np.mean(h):.2f} ({sum(h)}/{len(h)})"
              + ("  [item-only bootstrap, seeds fixed: comparison, not used]" if k.endswith("_item") else ""))
    if "4" in args.sections:
        rng = np.random.default_rng(1515 + 4)  # independent stream per section (sections may run in separate jobs)
        dry_synth_16(rng, args)
    if "5" in args.sections:
        rng = np.random.default_rng(1515 + 5)  # independent stream per section (sections may run in separate jobs)
        dry_synth_p1(rng)


def synth_16(rng, b_id=0.5, b_ood=0.5, b_pert=None, u8=False):
    """GSM8K (1319, k=4, U 4x4) + GSM-Plus (300 seeds x 7, k=2, U one k=2 evaluation) + SVAMP (1000, k=4) + MATH
    (600, k=4, harder); world 'sym' per set; GSM-Plus items are perturbations of 300 of the GSM8K items."""
    b_pert = b_ood if b_pert is None else b_pert
    arms = ("gold_250", "gold_500", "teacher_250", "teacher_500")
    cells = {}
    th8 = rng.normal(-0.15, 2.3, 1319)
    dl8 = {"gold": rng.normal(-0.7, 1.3, 1319), "teacher": rng.normal(0.5, 1.3, 1319)}
    si = np.arange(300)
    specs = {"gsm8k": (th8, dl8, 4, 4, b_id),
             "gsm_plus": (np.repeat(th8[si], 7) + rng.normal(0, 0.7, 2100),
                          {k: np.repeat(v[si], 7) + rng.normal(0, 0.5, 2100) for k, v in dl8.items()}, 2, 4 if u8 else 1, b_pert),
             "svamp": (rng.normal(0.3, 2.3, 1000), {"gold": rng.normal(-0.5, 1.2, 1000), "teacher": rng.normal(0.4, 1.2, 1000)}, 4, 2 if u8 else 1, b_ood),
             "math600": (rng.normal(-1.8, 2.0, 600), {"gold": rng.normal(-0.5, 1.0, 600), "teacher": rng.normal(0.3, 1.0, 600)}, 4, 2 if u8 else 1, b_ood)}
    for st, (th, dl, k, nU, b0) in specs.items():
        N = len(th)
        pU = L.expit(th)
        cells[f"{st}:U"] = _binom_units(rng, f"{st}:U", None, [pU] * nU, k, st, seeds=range(nU))
        cells[f"{st}:U"].lineage = None
        qR = [L.expit(th + 1.0 + rng.normal(0, 0.5, N)) for _ in SEEDS]
        cells[f"{st}:grpo|375"] = _binom_units(rng, f"{st}:grpo|375", "grpo", qR, k, st)
        for arm in arms:
            src, n = arm.split("_")
            delta = dl[src] + (rng.normal(0, 0.5, N) if n == "500" else 0)
            ps = [L.expit(th + delta + rng.normal(0, 0.15, N)) for _ in SEEDS]
            qs = [np.clip(qR[i] + b0 * (ps[i] - pU), 0.0, 1.0) for i in range(3)]
            cells[f"{st}:{arm}|pre"] = _binom_units(rng, f"{st}:{arm}|pre", arm, ps, k, st)
            cells[f"{st}:{arm}|375"] = _binom_units(rng, f"{st}:{arm}|375", arm, qs, k, st)
    clusters = [{"gsm8k": np.array([j]), "gsm_plus": np.arange(7 * j, 7 * j + 7)} for j in si]
    seed_mask = np.zeros(1319, bool)
    seed_mask[si] = True
    d = build_16_design(cells, {"gsm8k": 1319, "gsm_plus": 2100, "svamp": 1000, "math600": 600}, clusters, seed_mask, np.ones(2100, bool))
    return d, seed_mask, np.ones(2100, bool)


def dry_synth_16(rng, args):
    P(f"\n## 4. prereg 16 synthetic: O1 (OOD - ID retention) and O2 (perturbed - seeds); {args.reps16} datasets, B={args.cov_B}")
    abs_ = {"gold": POOLS["gold_pool"], "teacher": POOLS["teacher_pool"]}
    for lab, kw in (("null (all b = 0.5)", {}), ("O1 alternative (all OOD sets b 0.75)", {"b_ood": 0.75}),
                    ("O2 alternative (b_pert 0.65, other sets 0.5)", {"b_pert": 0.65}),
                    ("null, untrained k=8 on OOD sets", {"u8": True}), ("O1 alternative, untrained k=8", {"b_ood": 0.75, "u8": True}),
                    ("O2 alternative, untrained k=8", {"b_pert": 0.65, "u8": True})):

        res = defaultdict(list)
        for _ in range(args.reps16):
            d, sm, hs = synth_16(rng, **kw)
            gate = {(st, a): True for st in OOD16 for a in ARMS_A}
            obs, reps = L.bootstrap(d, stats16_fn(abs_, gate, sm, hs, None, args.tau, graders=("len",), with_types=False),
                                    args.cov_B, seed=int(rng.integers(1 << 30)))
            for src in abs_:
                k = f"len|O1|{src}|delta"
                lo, hi = L.ci(reps[k])
                res[("O1", src)].append((obs[k], lo, hi, obs[k] >= REG["O1_MIN"] and lo > 0))
            lo, hi = L.ci(reps["len|O2|delta"])
            res[("O2", "pooled")].append((obs["len|O2|delta"], lo, hi, obs["len|O2|delta"] >= REG["O2_MIN"] and lo > 0))
        for (o, src), rr in res.items():
            a = np.array(rr, float)
            P(f"  {lab}: {o} {src}: mean delta {a[:, 0].mean():+.3f} (SD {a[:, 0].std():.3f}), mean CI width "
              f"{np.mean(a[:, 2] - a[:, 1]):.3f}, P(registered rule holds) {a[:, 3].mean():.2f}")


def dry_synth_p1(rng):
    P("\n## 5. Part 1 synthetic (600 items, 16 samples per set and start; planted mechanism vs none; B=200)")
    for lab, strength in (("planted", 1.0), ("null", 0.0)):
        N, K = 600, 16
        th = rng.normal(0, 2.0, N)
        rev = rng.normal(0, 1, N)  # item-level RL propensity beyond difficulty and D_SFT (drives Delta and the realised change)
        rows = {"A": [], "B": [], "C": []}
        p0 = L.expit(th)
        for st in ("untrained", "gold_mf_s42", "gold_mf_s43", "gold_mf_s44"):
            dS = 0 if st == "untrained" else rng.normal(-0.5, 1.0, N)
            p = L.expit(th + dS)
            dsft = p - p0
            mech = strength * (-3 * dsft + 0.5 * rev) if st != "untrained" else 0.1 * th
            for s_ in "AB":
                for i in range(N):
                    for k in range(K):
                        c = bool(rng.random() < p[i])
                        base = {"item": i, "sample": k, "start": st, "correct_lenient": c, "correct_marked": c}
                        if s_ == "B":
                            rows["B"].append(base)
                            continue
                        f0 = rng.normal(0, 1) + (mech[i] if c else 0.0)
                        rows["A"].append({**base, "direction_name": "v0", "phi": f0})
                        for dn in ("rand_0", "rand_1"):
                            rows["A"].append({**base, "direction_name": dn, "phi": rng.normal(0, 1)})
                        rows["A"].append({**base, "direction_name": "v150", "phi": 0.6 * f0 + rng.normal(0, 0.8)})
                        rows["A"].append({**base, "direction_name": "v25", "phi": 0.3 * f0 + rng.normal(0, 0.3 if c else 0.6)})
            q = L.expit(th + 1.0 + (dS - strength * (2 * dsft - 0.8 * rev) if st != "untrained" else 0))
            for name in ([f"post_untrained_s{s}" for s in (42, 43, 44)] if st == "untrained" else [f"post_{st}"]):
                for i in range(N):
                    for k in range(K):
                        c = bool(rng.random() < q[i])
                        rows["C"].append({"item": i, "sample": k, "start": name, "correct_lenient": c, "correct_marked": c})
        P(f"\n  --- {lab} world")
        run_part1(rows, [], B=200)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["0", "A", "B", "X", "1", "16", "all"])
    ap.add_argument("--dry_run", choices=["power", "power_check", "synthetic", "e2"])
    ap.add_argument("--procs", type=int, default=4, help="processes for --dry_run power_check")
    ap.add_argument("--schema_only", action="store_true")
    ap.add_argument("--unblind", action="store_true", help="required to compute outcome statistics on blinded data")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--px_template", default="runs/gsm8k/Llama-3.2-1B-Instruct-px-{proto}{src}")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--nsim", type=int, default=200)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--p1_dir", default=None)
    ap.add_argument("--c_phi", type=float, default=0.5, help="cost of phi per sample relative to a generation (Part 1 crossover)")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--null_reps", type=int, default=4)
    ap.add_argument("--cov_reps", type=int, default=30)
    ap.add_argument("--cov_B", type=int, default=200)
    ap.add_argument("--power_reps", type=int, default=200)
    ap.add_argument("--reps16", type=int, default=30)
    ap.add_argument("--sections", default="12345", help="synthetic dry-run sections to run")
    args = ap.parse_args()
    paths = Paths(args.root, args.px_template)
    P(f"# score_prereg15.py args={vars(args)}\n# tau={args.tau:.4f} REG={REG}")
    t0 = time.time()
    if args.dry_run == "e2":
        dry_e2(paths, args)
    elif args.dry_run == "power_check":
        dry_power_check(args)
    elif args.dry_run == "power":
        dry_power(args)
    elif args.dry_run == "synthetic":
        dry_synthetic(args)
    elif args.schema_only:
        schema(paths, args.part)
    else:
        if args.part in ("0", "A", "B", "X", "16", "all") and not args.unblind:
            sys.exit("blinded data: pass --unblind only after prereg 15/16 are registered (committed and pushed)")
        res = {}
        if args.part in ("0", "all"):
            res["0"] = part_0(paths, args)
        if args.part in ("A", "all"):
            res["A"] = part_A(paths, args)
        if args.part in ("B", "all"):
            part_B(paths, args)
        if args.part in ("16", "all"):
            res["16"] = part_16(paths, args)
        if args.part in ("X", "all"):
            res["X"] = part_X(paths, args)
        if args.part in ("1", "all"):
            if args.part == "all" and not res.get("X", {}).get("X1"):
                P("\nPart 1 conditional rule: X1 has not held (lenient) -> Part 1 is not run")
            else:
                part_1(args)
    P(f"\n# done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
