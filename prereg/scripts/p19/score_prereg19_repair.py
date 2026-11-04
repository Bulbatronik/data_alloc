#!/usr/bin/env python3
"""Score prereg/19 M1, "repair needs signal" (per-item and arm-level). Written and dry-run BEFORE registration: no
statistic of this file has been computed on real data (MECH_PLAN.md M1; blinding rules of the T19 brief).

Imports the FROZEN prereg-15/17 libraries by path (prereg/scripts/p17/{p15lib,score_prereg15,score_prereg17}.py; SHA-256
prefixes printed) and reuses, unchanged: the Part A loaders and registered exclusions (S.load_A, S.part_A_exclusions,
S.cells_A), the 17B loader and cells (S17.part_17b, intercepted at its bootstrap call as score_prereg18.py does; the
stop-rate exclusion is switched off for the post hoc all-runs variant exactly as lead_p17b_noexcl.py does), the
hierarchical bootstrap (L.bootstrap: seeds within lineage, then items), the graders (lenient primary, marked secondary)
and the one-index tilt null (L.Latent2D + L.fit_tilt, simulated through identical cells with L.binom_cell).

PER-ITEM STATISTIC (probability scale; per arm quadruple (S, U, PS, PR) = SFT start pre-RL, untrained, SFT start post-RL,
RL alone post-RL). Cross-fitted over the two sample halves f in {0, 1} (half 0 = samples < k//2 of every evaluation):
  selection on half f:   hurt  H_f = { p_S^f - p_U^f <= -tau }                     (tau = 1/3, as prereg 15/17/18)
                         zero-signal Z_f = H_f and p_S^f = 0 ; signal N_f = H_f and p_S^f > 0
  measurement on half m = 1 - f and on ALL post-RL samples (independent of the selection: no regression to the mean):
                         damage  dmg_i = p_U^m - p_S^m ,   repair  rep_i = q_S - p_S^m   (Delta_S of the brief)
  per group g in {Z, N}: rho_g = sum_f sum_{i in g_f} rep_i / sum_f sum_{i in g_f} dmg_i   (share of the damage the SFT
                         start's own movement repairs; ratio of item sums, pooled over arms by summing)
                         rep_g_pts = mean repair per item (points)
  contrast               C = rho_N - rho_Z            (C_pts = rep_N_pts - rep_Z_pts reported)
  null                   one-index tilt fitted on the untrained start's own pair (q_R = expit(logit p_U + s_U)) and
                         applied to BOTH starts (q_S = expit(logit p_S + s_U)): the registered 15/17 null. Under it an
                         item with p_S ~ 0 cannot move, so C_null > 0 (location effect); 'own' sensitivity null: the
                         SFT start's own shift s_S fitted on its own pair (a memoryless, signal-proportional repair of
                         the observed size).
  excess                 X_ex = X - median(X | tilt null), X in {rho_Z, rho_N, C}. C_ex > 0: the repair beyond the
                         common RL shift concentrates on items where the damaged start still samples correct answers.
  composition (Gemma)    w_Z = sum dmg_Z / sum dmg_H; rho_H = w_Z rho_Z + (1 - w_Z) rho_N. Gemma (gold + teacher, all
                         runs) vs Llama gold (17B, same items): two-fold decomposition of the gap in excess repair
                         d = rho_H^ex(Llama) - rho_H^ex(Gemma) = [composition (w_Z^G - w_Z^L) * mean_sets(rho_N^ex -
                         rho_Z^ex)] + [rates]; kappa = composition / d (share of Gemma's lower repair explained by its
                         larger zero-signal share).
ARM-LEVEL STATISTIC: per training run, the share of zero-variance GRPO groups (all 8 rewards equal) in the first 50 steps
  (rollouts.jsonl; the S' GSM8K runs predate --log_rollouts, so TRL's logged frac_reward_zero_std is used, entries at
  steps 10-50); x_arm = mean over the arm's runs - mean over the family's RL-alone runs. Repair y_arm = test-set gap
  closed at the final step, 1 - mean(q_S - q_R) / mean(p_S - p_U) (lenient). Damaged arms (a priori, already-scored
  sign of the arm-level gap): S' gold_50/250/500, 17B Llama gold, Gemma gold, Gemma teacher.

Usage (Slurm CPU only; never on the login node):
  python -I runs/theory/scratch/score_prereg19_repair.py --schema_only
  python -I runs/theory/scratch/score_prereg19_repair.py --dry_run synthetic [--reps 100 --procs 16 --calib_out f.jsonl]
  python -I runs/theory/scratch/score_prereg19_repair.py --dry_run aggregate --calib_in f.jsonl
  python -I runs/theory/scratch/score_prereg19_repair.py --dry_run arm            # arm-level rule calibration
  python -I runs/theory/scratch/score_prereg19_repair.py --dry_run mock --mock_root <new dir>   # unblind path, synthetic files
  python -I runs/theory/scratch/score_prereg19_repair.py --unblind               # refused unless prereg/19_*.md exists
"""
import argparse
import glob
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FROZEN = REPO / "prereg" / "scripts" / "p17"
sys.path.insert(0, str(FROZEN))
import p15lib as L  # noqa: E402
import score_prereg15 as S  # noqa: E402
import score_prereg17 as S17  # noqa: E402

for _m in (L, S, S17):
    assert Path(_m.__file__).resolve().parent == FROZEN, f"{_m.__name__} not imported from {FROZEN}"

P, fmt, med = S.P, S.fmt, S.med
NAN = float("nan")
TAU = S.TAU

# ---------------------------------------------------------------- constants (PROPOSED; the lead fixes them at registration)
REG19 = dict(
    TAU=S.TAU, B=2000, NSIM=200,
    MIN_N=30,            # gate: >= 30 items per arm in BOTH Z and N (selection half, mean over folds)
    MIN_DMG_PTS=10.0,    # gate: mean measured damage >= 10 points in both groups
    T_C=0.10,            # M1-1: C_ex > T_C (calibrated: see T19 report section 3; set from --dry_run aggregate)
    T_RATIO=0.5,         # M1-2 (reported unless calibrated): rho_Z^ex <= T_RATIO * rho_N^ex with rho_N^ex > 0
    KAPPA=0.5, MIN_D=0.10,  # M1-3: kappa >= 0.5, gated on d >= 0.10
    EARLY=50,            # arm-level window: first 50 GRPO steps
    SPEAR=-0.6,          # A-1: Spearman(x, y) over the damaged arms <= -0.6
    SEEDS=dict(boot=1901, null=1902, arm=1903),
)
PRIMARY_SETS = ("gsm8k:gold_pool", "17b:llama_gold", "17b:gemma_harm")
DAMAGED = ("S:gold_50", "S:gold_250", "S:gold_500", "llama:gold", "gemma:gold", "gemma:teacher")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]


# ================================================================ per-item estimator
GK = ("n", "rep", "dmg", "conv", "ps", "pu", "qs", "qr")


def repair_parts(v, o, quad, tau=TAU):
    """Cross-fitted sums for Z (hurt, p_S^f = 0) and N (hurt, p_S^f > 0); n is the mean count per fold, every other key
    is summed over both folds (per-item mean = sum / (2 n))."""
    S_, U_, PS, PR = quad
    ps0, ps1, _ = v.m(S_, o, "p")
    pu0, pu1, _ = v.m(U_, o, "p")
    qs, qr = v.m(PS, o, "p")[2], v.m(PR, o, "p")[2]
    out = {f"{g}_{k}": 0.0 for g in "ZN" for k in GK}
    for f in (0, 1):
        sS, sU, mS, mU = (ps0, pu0, ps1, pu1) if f == 0 else (ps1, pu1, ps0, pu0)
        hurt = sS - sU <= -tau + 1e-9
        zero = sS <= 1e-12
        for g, mask in (("Z", hurt & zero), ("N", hurt & ~zero)):
            if not mask.any():
                continue
            out[f"{g}_n"] += mask.sum() / 2
            out[f"{g}_rep"] += float((qs - mS)[mask].sum())
            out[f"{g}_dmg"] += float((mU - mS)[mask].sum())
            out[f"{g}_conv"] += float((qr - mU)[mask].sum())
            out[f"{g}_ps"] += float(mS[mask].sum())
            out[f"{g}_pu"] += float(mU[mask].sum())
            out[f"{g}_qs"] += float(qs[mask].sum())
            out[f"{g}_qr"] += float(qr[mask].sum())
    return out


def repair_from(Pp, n_arms=1):
    r = {}
    for g in "ZN":
        n, dmg = Pp[f"{g}_n"], Pp[f"{g}_dmg"]
        ok = n > 0 and dmg > 1e-9
        r[f"n_{g}"] = n / n_arms
        r[f"rho_{g}"] = Pp[f"{g}_rep"] / dmg if ok else NAN
        r[f"conv_{g}"] = Pp[f"{g}_conv"] / dmg if ok else NAN          # RL alone's movement / damage (reported)
        r[f"closed_{g}"] = 1 - (Pp[f"{g}_qr"] - Pp[f"{g}_qs"]) / dmg if ok else NAN  # post gap closed on the group
        for k in ("rep", "dmg", "ps", "pu", "qs", "qr"):
            r[f"{k}_{g}_pts"] = 100 * Pp[f"{g}_{k}"] / (2 * n) if n > 0 else NAN
    r["C"] = r["rho_N"] - r["rho_Z"]
    r["C_pts"] = r["rep_N_pts"] - r["rep_Z_pts"]
    dH = Pp["Z_dmg"] + Pp["N_dmg"]
    r["wZ"] = Pp["Z_dmg"] / dH if dH > 1e-9 else NAN
    r["rho_H"] = (Pp["Z_rep"] + Pp["N_rep"]) / dH if dH > 1e-9 else NAN
    return r


STAT_KEYS = tuple(repair_from({f"{g}_{k}": 0.0 for g in "ZN" for k in GK}))


def set_stats(v, o, quads):
    return repair_from(L.add_parts([repair_parts(v, o, q) for q in quads]), len(quads))


def gap_closed(v, o, quad):
    """Arm-level test-set gap closed: 1 - mean(q_S - q_R) / mean(p_S - p_U) (all samples, probability scale)."""
    S_, U_, PS, PR = quad
    pre = float(np.mean(v.m(S_, o)[2] - v.m(U_, o)[2]))
    post = float(np.mean(v.m(PS, o)[2] - v.m(PR, o)[2]))
    return (1 - post / pre if abs(pre) > 1e-9 else NAN), 100 * pre, 100 * post


def gate(r):
    return (r["n_Z"] >= REG19["MIN_N"] and r["n_N"] >= REG19["MIN_N"] and r["dmg_Z_pts"] >= REG19["MIN_DMG_PTS"]
            and r["dmg_N_pts"] >= REG19["MIN_DMG_PTS"])


# ---------------------------------------------------------------- tilt nulls through the identical cells
def tilt_fit(d, o, quad):
    S_, U_, PS, PR = quad
    cU, nU = L.counts(d.cells[U_], o)
    cS, nS = L.counts(d.cells[S_], o)
    cq, nq = L.counts(d.cells[PS], o)
    cr, nr = L.counts(d.cells[PR], o)
    lat = L.Latent2D(cU, nU, cS, nS)
    return lat, L.fit_tilt(lat, lat.A, cr, nr), L.fit_tilt(lat, lat.B, cq, nq)


def null_parts(rng, d, o, quad, lat, sU, sS, nsim, kinds=("tilt", "own")):
    S_, U_, PS, PR = quad
    out = {k: [] for k in kinds}
    for _ in range(nsim):
        a, b = lat.draw(rng)
        pU, pS, qR = L.expit(a), L.expit(b), L.expit(a + sU)
        for k in kinds:
            qS = L.expit(b + (sU if k == "tilt" else sS))
            cells = {U_: L.binom_cell(rng, d.cells[U_], pU), S_: L.binom_cell(rng, d.cells[S_], pS),
                     PS: L.binom_cell(rng, d.cells[PS], qS), PR: L.binom_cell(rng, d.cells[PR], qR)}
            out[k].append(repair_parts(L.View(L.Design(cells)), "sim", quad))
    return out


def pool_null(per_arm, n_arms):
    """per_arm: list (one per arm) of lists of parts; replicate r pooled over arms -> arrays of derived stats."""
    acc = defaultdict(list)
    for parts in zip(*per_arm):
        for k, x in repair_from(L.add_parts(list(parts)), n_arms).items():
            acc[k].append(x)
    return {k: np.array(v, float) for k, v in acc.items()}


EX_KEYS = ("rho_Z", "rho_N", "C", "C_pts", "rho_H")


def excess(r, nl):
    return {f"{k}_ex": r[k] - med(nl[k]) for k in EX_KEYS}


def composition(G, Lm):
    """G, Lm: dicts with rho_Z_ex, rho_N_ex, wZ (Gemma pool, Llama gold). Returns (kappa, composition part, d)."""
    rH = lambda x: x["wZ"] * x["rho_Z_ex"] + (1 - x["wZ"]) * x["rho_N_ex"]
    d = rH(Lm) - rH(G)
    gbar = 0.5 * ((G["rho_N_ex"] - G["rho_Z_ex"]) + (Lm["rho_N_ex"] - Lm["rho_Z_ex"]))
    comp = (G["wZ"] - Lm["wZ"]) * gbar
    return (comp / d if np.isfinite(d) and d >= REG19["MIN_D"] else NAN), comp, d


# ================================================================ analysis shared by --unblind and --dry_run mock
def analyse_items(d, sets, graders, B, nsim, bseed, nseed, label, armsteps=None):
    """sets: {set name: [quad, ...]} (one design). Returns {(o, set): {'obs', 'rep', 'null', 'ex'}}; prints tables."""
    def fn(v):
        out = {}
        for o in graders:
            for s, qs in sets.items():
                for k, x in set_stats(v, o, qs).items():
                    out[f"{o}|{s}|{k}"] = x
            for a, q in (armsteps or {}).items():
                g, pre, post = gap_closed(v, o, q)
                out[f"{o}|G|{a}"], out[f"{o}|Gpre|{a}"], out[f"{o}|Gpost|{a}"] = g, pre, post
        return out
    t0 = time.time()
    obs, reps = L.bootstrap(d, fn, B, seed=bseed)
    P(f"\n# {label}: bootstrap B = {B} in {time.time() - t0:.0f}s")
    rng = np.random.default_rng(nseed)
    res = {}
    for o in graders:
        fits, sims = {}, {}
        for s, qs in sets.items():
            for q in qs:
                if q not in fits:
                    t1 = time.time()
                    lat, sU, sS = tilt_fit(d, o, q)
                    fits[q] = (sU, sS)
                    sims[q] = null_parts(rng, d, o, q, lat, sU, sS, nsim)
                    P(f"   null fit [{o}] {q[0]}: tilt s_U {sU:+.3f} (registered null), own shift s_S {sS:+.3f} "
                      f"({time.time() - t1:.0f}s)")
        for s, qs in sets.items():
            nl = {k: pool_null([sims[q][k] for q in qs], len(qs)) for k in ("tilt", "own")}
            r = {k: obs[f"{o}|{s}|{k}"] for k in STAT_KEYS}
            rr = {k: reps[f"{o}|{s}|{k}"] for k in r}
            ex = excess(r, nl["tilt"])
            exo = {k.replace("_ex", "_exown"): r[k[:-3]] - med(nl["own"][k[:-3]]) for k in ex}
            res[(o, s)] = {"obs": r, "rep": rr, "null": nl, "ex": ex, "exown": exo, "gate": gate(r)}
            P(f"\n  [{o}] {s}  ({len(qs)} arm(s): {', '.join(q[0] for q in qs)})  gate {'PASS' if gate(r) else 'FAIL'}")
            P(f"    items per arm: Z {r['n_Z']:.1f}, N {r['n_N']:.1f}; damage (pts) Z {r['dmg_Z_pts']:.1f}, N {r['dmg_N_pts']:.1f}; "
              f"w_Z {r['wZ']:.3f}")
            for k in ("rho_Z", "rho_N", "C", "C_pts", "rho_H", "conv_Z", "conv_N", "closed_Z", "closed_N", "rep_Z_pts", "rep_N_pts"):
                line = f"    {k:10s} {fmt(r[k], L.ci(rr[k]))}"
                if k in EX_KEYS:
                    lo, hi = L.ci(rr[k])
                    m0, m1 = med(nl["tilt"][k]), med(nl["own"][k])
                    line += (f" | tilt null median {m0:+.3f} -> excess {fmt(r[k] - m0, (lo - m0, hi - m0))}"
                             f" | own-shift null {m1:+.3f} -> {r[k] - m1:+.3f}")
                P(line)
    return obs, reps, res


def decide(res, o="len"):
    P(f"\n## PROPOSED registered rules [{o}] (thresholds from the synthetic calibration; see the T19 report)")
    out = {}
    for s in PRIMARY_SETS:
        x = res.get((o, s))
        if x is None:
            P(f"  M1-1 {s}: not formable")
            continue
        r, ex = x["obs"], x["ex"]
        held = x["gate"] and ex["C_ex"] > REG19["T_C"]
        lo, hi = L.ci(x["rep"]["C"])
        m0 = med(x["null"]["tilt"]["C"])
        P(f"  M1-1 {s}: C_ex {ex['C_ex']:+.3f} [{lo - m0:+.3f}, {hi - m0:+.3f}] vs > {REG19['T_C']} (gate "
          f"{'pass' if x['gate'] else 'FAIL -> untestable'}) -> {'HELD' if held else ('not held' if x['gate'] else 'UNTESTABLE')}")
        rz, rn = ex["rho_Z_ex"], ex["rho_N_ex"]
        P(f"  M1-2 {s} (reported): rho_Z^ex {rz:+.3f}, rho_N^ex {rn:+.3f}; rho_Z^ex <= {REG19['T_RATIO']} rho_N^ex -> "
          f"{'yes' if rn > 0 and rz <= REG19['T_RATIO'] * rn else 'no'}")
        out[s] = held if x["gate"] else None
    return out


def decide_composition(res, o="len"):
    G, Lm = res.get((o, "17b:gemma_harm")), res.get((o, "17b:llama_gold"))
    if G is None or Lm is None:
        P("  M1-3 composition: not formable")
        return None
    gx = dict(G["ex"], wZ=G["obs"]["wZ"])
    lx = dict(Lm["ex"], wZ=Lm["obs"]["wZ"])
    kap, comp, d = composition(gx, lx)
    # bootstrap: same replicate index (joint 17B design: items resampled jointly)
    m0 = {k: med(G["null"]["tilt"][k]) for k in ("rho_Z", "rho_N")}
    m1 = {k: med(Lm["null"]["tilt"][k]) for k in ("rho_Z", "rho_N")}
    ks = []
    for i in range(len(G["rep"]["rho_Z"])):
        g_ = {"rho_Z_ex": G["rep"]["rho_Z"][i] - m0["rho_Z"], "rho_N_ex": G["rep"]["rho_N"][i] - m0["rho_N"], "wZ": G["rep"]["wZ"][i]}
        l_ = {"rho_Z_ex": Lm["rep"]["rho_Z"][i] - m1["rho_Z"], "rho_N_ex": Lm["rep"]["rho_N"][i] - m1["rho_N"], "wZ": Lm["rep"]["wZ"][i]}
        ks.append(composition(g_, l_)[0])
    held = np.isfinite(kap) and kap >= REG19["KAPPA"]
    P(f"  M1-3 [{o}] Gemma persistence by composition: w_Z Gemma {gx['wZ']:.3f} vs Llama {lx['wZ']:.3f}; d (excess repair, "
      f"Llama - Gemma) {d:+.3f}; composition part {comp:+.3f}; kappa {fmt(kap, L.ci(ks))} vs >= {REG19['KAPPA']} "
      f"(gate d >= {REG19['MIN_D']}) -> " + ("HELD" if held else ("not held" if np.isfinite(kap) else "UNTESTABLE")))
    return held


# ================================================================ arm level: zero-variance shares
def rollout_groups(path):
    """rollouts.jsonl -> {step: [(n rewards, all equal, all zero, all one), ...]}; resume replays removed as in
    S.rollouts (a step that goes back marks a replay; keep the last pass); a (step, example_id) block is split into
    groups of 8 in file order; censored / None rewards are dropped from the group."""
    rows = []
    for x in L.read_jsonl(path):
        x = {"step": x["step"], "example_id": x.get("example_id"), "reward": x.get("reward"), "censored": x.get("censored")}
        if rows and x["step"] < rows[-1]["step"]:
            rows = [y for y in rows if y["step"] < x["step"]]
        rows.append(x)
    blocks = defaultdict(list)
    for x in rows:
        blocks[(x["step"], x["example_id"])].append(None if x["censored"] else x["reward"])
    out = defaultdict(list)
    bad = 0
    for (st, _), rw in blocks.items():
        if len(rw) % 8:
            bad += 1
        for j in range(0, len(rw), 8):
            g = [float(r) for r in rw[j:j + 8] if r is not None]
            if len(g) >= 2:
                out[st].append((len(g), max(g) - min(g) < 1e-9, max(g) < 1e-9, min(g) > 1 - 1e-9))
    return out, bad


def zero_share_rollouts(path, first=REG19["EARLY"]):
    groups, bad = rollout_groups(path)
    steps = sorted(groups)
    early = steps[:first]
    g = [x for st in early for x in groups[st]]
    gall = [x for st in steps for x in groups[st]]
    f = lambda gs, j: float(np.mean([x[j] for x in gs])) if gs else NAN
    return {"zv": f(g, 1), "zero": f(g, 2), "one": f(g, 3), "zv_all": f(gall, 1), "zero_all": f(gall, 2),
            "n_steps": len(steps), "n_groups_early": len(g), "bad_blocks": bad, "src": "rollouts"}


def zero_share_trainer_state(path, first=REG19["EARLY"]):
    lh = [x for x in json.load(open(path))["log_history"] if "frac_reward_zero_std" in x]
    early = [x["frac_reward_zero_std"] for x in lh if x["step"] <= first]
    return {"zv": float(np.mean(early)) if early else NAN, "zero": NAN, "one": NAN,
            "zv_all": float(np.mean([x["frac_reward_zero_std"] for x in lh])), "zero_all": NAN,
            "n_steps": max(x["step"] for x in lh), "n_groups_early": len(early), "bad_blocks": 0, "src": "trainer_state"}


def arm_runs(paths):
    """{arm: {seed: path}} for every arm with a zero-variance source. S' GSM8K: trainer_state (no rollouts); 17B: rollouts."""
    runs = {}
    for arm in S.ARMS_A + ("grpo",):
        src = "teacher" if arm.startswith("teacher") else "gold"
        meth = "full_grpo" if arm == "grpo" else f"random_sft_{arm.split('_')[1]}"
        runs[f"S:{arm}"] = {s: paths.sprime[src] / f"seed_{s}/{meth}/grpo/checkpoint-375/trainer_state.json" for s in S.SEEDS}
    for fam in S17.FAMILIES:
        for arm in ("grpo",) + S17.ARMS17B:
            meth = "full_grpo" if arm == "grpo" else "listed_sft"
            runs[f"{fam}:{arm}"] = {s: S17.run_dir17b(paths.root, fam, "mf", arm) / f"seed_{s}/{meth}/grpo/rollouts.jsonl"
                                    for s in S17.SEEDS17B[arm]}
    return runs


def comparator(arm):
    return "S:grpo" if arm.startswith("S:") else arm.split(":")[0] + ":grpo"


def arm_level(paths, G_obs, G_rep, o="len", seed=REG19["SEEDS"]["arm"], excl=None):
    """G_obs: {arm: gap closed}; G_rep: {arm: bootstrap array}. Prints the arm table and the proposed rules."""
    runs = arm_runs(paths)
    z = {}
    for arm, rs in runs.items():
        z[arm] = {}
        for s, f in rs.items():
            if excl and (arm, s) in excl:
                continue
            if not Path(f).exists():
                continue
            z[arm][s] = zero_share_trainer_state(f) if f.name == "trainer_state.json" else zero_share_rollouts(f)
    P(f"\n## Arm level [{o}]: zero-variance share of GRPO groups in the first {REG19['EARLY']} steps (per run), test gap closed")
    for arm in runs:
        for s, r in sorted(z[arm].items()):
            P(f"  {arm:16s} s{s}: zv {r['zv']:.3f}  all-wrong {r['zero']:.3f}  all-right {r['one']:.3f}  zv(all steps) "
              f"{r['zv_all']:.3f}  [{r['src']}, steps {r['n_steps']}, early groups/entries {r['n_groups_early']}, "
              f"blocks not multiple of 8: {r['bad_blocks']}]")
    xs, xr = {}, {}
    rng = np.random.default_rng(seed)
    nb = len(next(iter(G_rep.values()))) if G_rep else 0
    for arm in runs:
        if arm.endswith(":grpo") or not z[arm] or not z.get(comparator(arm)):
            continue
        a = np.array([r["zv"] for r in z[arm].values()])
        c = np.array([r["zv"] for r in z[comparator(arm)].values()])
        xs[arm] = a.mean() - c.mean()
        xr[arm] = np.array([rng.choice(a, len(a)).mean() - rng.choice(c, len(c)).mean() for _ in range(nb)])
    P("\n  arm               x = zv(arm) - zv(RL alone)    y = gap closed (final step)")
    for arm in xs:
        P(f"  {arm:16s} {fmt(xs[arm], L.ci(xr[arm]))}    {fmt(G_obs.get(arm, NAN), L.ci(G_rep.get(arm, [])))}"
          + ("   [damaged]" if arm in DAMAGED else ""))
    D = [a for a in DAMAGED if a in xs and a in G_obs]
    if len(D) >= 4:
        sp = S17.spearman([xs[a] for a in D], [G_obs[a] for a in D])
        spr = [S17.spearman([xr[a][i] for a in D], [G_rep[a][i] for a in D]) for i in range(nb)]
        top = sorted(D, key=lambda a: -xs[a])[:2]
        P(f"  A-1 Spearman(x, y) over damaged arms {D}: {fmt(sp, L.ci(spr))} vs <= {REG19['SPEAR']} -> "
          + ("HELD" if sp <= REG19["SPEAR"] else "not held"))
        P(f"  A-2 the two Gemma arms have the largest x among damaged arms: top-2 = {top} -> "
          + ("HELD" if set(top) == {"gemma:gold", "gemma:teacher"} else "not held"))
        allA = [a for a in xs if a in G_obs]
        P(f"  reported: Spearman over all {len(allA)} arms {S17.spearman([xs[a] for a in allA], [G_obs[a] for a in allA]):+.3f}")
    else:
        P(f"  A-1/A-2: only {len(D)} damaged arms formable; not scored")
    # validation of the substitution (S' arms use TRL's frac_reward_zero_std): both sources on runs that have both
    chk = []
    for arm, rs in runs.items():
        if arm.startswith("S:"):
            continue
        for s, f in rs.items():
            ts = Path(f).parent / f"checkpoint-{S17.STEPS17B[-1]}" / "trainer_state.json"
            if Path(f).exists() and ts.exists():
                a, b = zero_share_rollouts(f), zero_share_trainer_state(ts)
                chk.append((arm, s, a["zv"], b["zv"]))
    if chk:
        dd = [abs(a - b) for *_, a, b in chk]
        P(f"  substitution check (17B runs, rollouts vs trainer_state frac_reward_zero_std, early window): {len(chk)} runs, "
          f"max |diff| {max(dd):.3f}, mean |diff| {np.mean(dd):.3f}")
    return xs, xr


# ================================================================ real-data designs
class _Captured(Exception):
    pass


def capture_17b(paths, noexcl):
    """Designs of the frozen S17.part_17b (cells and, unless noexcl, the registered stop-rate exclusions)."""
    designs = []
    saved = (L.bootstrap, S.run_nulls, S17.decide17b, S.REG["STOP_MIN"])

    def boot(design, fn_, B, seed, seeds=True):
        designs.append(design)
        return {}, {}
    try:
        L.bootstrap = boot
        S.run_nulls = lambda *a, **k: ({}, {})
        S17.decide17b = lambda *a, **k: {}
        if noexcl:
            S.REG["STOP_MIN"] = -1.0  # as runs/reviews/round4/scratch/lead_p17b_noexcl.py (POST HOC for Gemma)
        S17.part_17b(paths, SimpleNamespace(B=0, nsim=0, proto="mf"))
    finally:
        L.bootstrap, S.run_nulls, S17.decide17b, S.REG["STOP_MIN"] = saved
    present = [f for f in S17.FAMILIES if (paths.root / S17.P17B / f"{f}_mf_untrained.jsonl").exists()]
    assert len(present) == len(designs)
    cells, ids0 = {}, None
    for fam, d in zip(present, designs):
        ids = d.cells["U"].ids
        ids0 = ids if ids0 is None else ids0
        assert ids == ids0, "17B families scored on different item lists"
        for c in d.cells.values():
            n = L.Cell(f"{fam}:{c.name}", c.set, None if c.lineage is None else f"{fam}:{c.lineage}", c.seeds, c.C, c.Nh)
            n.ids = getattr(c, "ids", None)
            cells[n.name] = n
    return L.Design(cells), present


def q17(cells, fam, arm, t=300):
    q = (f"{fam}:{arm}|pre", f"{fam}:U", f"{fam}:{arm}|{t}", f"{fam}:grpo|{t}")
    return q if all(c in cells for c in q) else None


def run_unblind(paths, args):
    res_all = {}
    # ---- GSM8K Part A (registered exclusions; step 375)
    ev, unt, e2u = S.load_A(paths)
    bad, rates = S.part_A_exclusions(paths)
    P("Part A excluded runs (registered stop-rate rule):", sorted(bad) or "none")
    dA = L.Design(S.cells_A(ev, unt, e2u, bad))
    qa = {a: S.quads([a], 375, cells=dA.cells) for a in S.ARMS_A}
    setsA = {"gsm8k:gold_pool": qa["gold_250"] + qa["gold_500"]}
    setsA.update({f"gsm8k:{a}": qa[a] for a in S.ARMS_A if qa[a]})
    armA = {f"S:{a}": qa[a][0] for a in S.ARMS_A if qa[a]}
    armA.update({f"S:{a}@{t}": S.quads([a], t, cells=dA.cells)[0] for a in S.ARMS_A for t in (75, 225) if S.quads([a], t, cells=dA.cells)})
    obsA, repA, resA = analyse_items(dA, setsA, L.GRADERS, args.B, args.nsim, REG19["SEEDS"]["boot"], REG19["SEEDS"]["null"],
                                     "GSM8K Part A (S', step 375)", armA)
    res_all.update(resA)
    # ---- 17B, all runs (Gemma POST HOC relative to 17B), one joint design
    dB, fams = capture_17b(paths, noexcl=True)
    setsB, armB = {}, {}
    for fam in fams:
        for arm in S17.ARMS17B:
            for t in (300, 150):
                q = q17(dB.cells, fam, arm, t)
                if q:
                    setsB[f"17b:{fam}_{arm}" + ("" if t == 300 else "@150")] = [q]
                    armB[f"{fam}:{arm}" + ("" if t == 300 else "@150")] = q
    gh =[q for q in (q17(dB.cells, "gemma", "gold"), q17(dB.cells, "gemma", "teacher")) if q]
    if gh:
        setsB["17b:gemma_harm"] = gh
    obsB, repB, resB = analyse_items(dB, setsB, L.GRADERS, args.B, args.nsim, REG19["SEEDS"]["boot"] + 10,
                                     REG19["SEEDS"]["null"] + 10, "17B MATH L3-5 (all runs; Gemma post hoc)", armB)
    res_all.update(resB)
    # ---- registered-exclusion variant (reported): which 17B arms keep >= 2 runs under the registered stop-rate rule
    excl = set()
    for fam in fams:
        for arm in ("grpo",) + S17.ARMS17B:
            meth = "full_grpo" if arm == "grpo" else "listed_sft"
            for s in S17.SEEDS17B[arm]:
                f = S17.run_dir17b(paths.root, fam, "mf", arm) / f"seed_{s}/{meth}/grpo/rollouts.jsonl"
                if f.exists() and S.stop_rate_rollouts(f)[0] < S.REG["STOP_MIN"]:
                    excl.add((f"{fam}:{arm}", s))
    keep = [f"{f}:{a}" for f in fams for a in S17.ARMS17B
            if sum((f"{f}:{a}", s) not in excl for s in S17.SEEDS17B[a]) >= 2]
    P(f"\n17B registered stop-rate exclusions: {sorted(excl) or 'none'}; arms with >= 2 runs under the registered rule: {keep}"
      " (the per-item sets above use all runs; Gemma is POST HOC relative to 17B)")
    for o in L.GRADERS:
        decide(res_all, o)
        decide_composition(res_all, o)
    # ---- arm level (lenient primary)
    for o in L.GRADERS:
        Go = {a: obsA[f"{o}|G|{a}"] for a in armA}
        Go.update({a: obsB[f"{o}|G|{a}"] for a in armB})
        Gr = {a: repA[f"{o}|G|{a}"] for a in armA}
        Gr.update({a: repB[f"{o}|G|{a}"] for a in armB})
        arm_level(paths, Go, Gr, o)
    return res_all


# ================================================================ synthetic calibration (real sizes)
SIZES = {
    # N items; untrained units x samples; SFT arms: seeds, k; RL alone seeds, k. Arm-level targets are the already
    # scored lenient values (Part A step 375 gold arms; 17B llama gold; 17B Gemma gold/teacher all runs, post hoc).
    "gsm8k": dict(N=1319, U=(4, 4), seeds=3, k=4, rl_seeds=3, pU=0.45, sd=2.0, rl_alone=0.048,
                  arms={"gold_250": dict(gap=-0.130, vd=0.056, rl_sft=0.106), "gold_500": dict(gap=-0.115, vd=0.053, rl_sft=0.083)}),
    "llama": dict(N=1000, U=(1, 24), seeds=2, k=8, rl_seeds=3, pU=0.179, sd=2.5, rl_alone=0.0675,
                  arms={"gold": dict(gap=-0.054, vd=0.022, rl_sft=0.128)}),
    "gemma": dict(N=1000, U=(1, 24), seeds=2, k=8, rl_seeds=3, pU=0.348, sd=2.5, rl_alone=0.021,
                  arms={"gold": dict(gap=-0.230, vd=0.094, rl_sft=0.107), "teacher": dict(gap=-0.191, vd=0.076, rl_sft=0.046)}),
}
SETS_SYN = {"gsm8k": {"gsm8k:gold_pool": ("gold_250", "gold_500")}, "llama": {"17b:llama_gold": ("gold",)},
            "gemma": {"17b:gemma_harm": ("gold", "teacher"), "17b:gemma_gold": ("gold",)}}
WORLDS = ("tilt", "uniform", "signal", "signal_hard", "half", "own", "signal_full")
SEED_SD, POST_SD = 0.3, 0.2


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def bisect(f, lo, hi, target, it=60):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


REPAIR_WORLDS = ("uniform", "signal", "signal_hard", "half", "signal_full")


def gfun(world, pS):
    """Repair gate: uniform 1; signal = P(a group of 8 samples is not all wrong) = 1 - (1 - p_S)^8; hard: p_S > 0.03;
    half: halfway between uniform and signal."""
    s = 1 - (1 - pS) ** 8
    return {"uniform": np.ones_like(pS), "signal": s, "signal_full": s, "signal_hard": (pS > 0.03).astype(float),
            "half": 0.5 + 0.5 * s}[world]


def world_qS(world, pS, pU, sR, par):
    """tilt: q_S = expit(logit p_S + s_R) (the null); own: expit(logit p_S + s_S) (memoryless shift of the observed size);
    repair worlds: the SFT start moves toward its pre-SFT profile p_U by omega * gate(p_S) (omega <= 1, no overshoot),
    then takes RL alone's common shift s_R plus an extra shift e >= 0 (only when omega = 1 cannot reach the arm's
    observed SFT-start movement). signal_full: omega = 1, e = 0 in every family (shared rates; composition check)."""
    if world == "tilt":
        return L.expit(logit(pS) + sR)
    if world == "own":
        return L.expit(logit(pS) + par["sS"])
    pt = np.clip(pS + par["omega"] * gfun(world, pS) * (pU - pS), 1e-4, 1 - 1e-4)
    return L.expit(logit(pt) + sR + par.get("extra", 0.0))


_TUNED = {}


def tune(size):
    """Prior and world parameters matched (on 200,000 draws, no sampling noise) to the size's arm-level targets."""
    if size in _TUNED:
        return _TUNED[size]
    Z = SIZES[size]
    r = np.random.default_rng(7)
    z, e = r.standard_normal(200000), r.standard_normal(200000)
    mU = bisect(lambda m: L.expit(m + Z["sd"] * z).mean(), -8, 8, Z["pU"])
    lu = mU + Z["sd"] * z
    pU = L.expit(lu)
    sR = bisect(lambda s: (L.expit(lu + s) - pU).mean(), -3, 6, Z["rl_alone"])
    arms = {}
    for a, A in Z["arms"].items():
        def mu_for(sig):
            return bisect(lambda m: (L.expit(lu + m + sig * e) - pU).mean(), -8, 4, A["gap"])

        def vd_for(sig):
            return np.var(L.expit(lu + mu_for(sig) + sig * e) - pU)
        sig = bisect(vd_for, 0.01, 6.0, A["vd"], it=30)
        mu = mu_for(sig)
        pS = L.expit(lu + mu + sig * e)
        par = {"mu": mu, "sig": sig, "achieved_vd": float(np.var(pS - pU))}
        par["sS"] = bisect(lambda s: (L.expit(logit(pS) + s) - pS).mean(), -3, 8, A["rl_sft"])
        for w in REPAIR_WORLDS:
            mv = lambda om, ex: float((world_qS(w, pS, pU, sR, {"omega": om, "extra": ex}) - pS).mean())
            if w == "signal_full":
                om, ex = 1.0, 0.0
            elif mv(1.0, 0.0) >= A["rl_sft"]:
                om, ex = bisect(lambda x: mv(x, 0.0), 0.0, 1.0, A["rl_sft"]), 0.0
            else:
                om, ex = 1.0, bisect(lambda x: mv(1.0, x), 0.0, 8.0, A["rl_sft"])
            par[f"omega_{w}"], par[f"extra_{w}"], par[f"achieved_{w}"] = om, ex, mv(om, ex)
        arms[a] = par
    _TUNED[size] = {"mU": mU, "sR": sR, "arms": arms}
    return _TUNED[size]


def cell_from(rng, name, lineage, probs, k, seeds):
    U, N = len(probs), len(probs[0])
    C = np.zeros((2, U, N))
    for u, p in enumerate(probs):
        C[0, u] = rng.binomial(k // 2, p)
        C[1, u] = rng.binomial(k - k // 2, p)
    Nh = np.array([[k // 2] * U, [k - k // 2] * U], float)
    return L.Cell(name, "syn", lineage, list(seeds), {"len": C}, Nh)


def synth(rng, size, world):
    Z, T = SIZES[size], tune(size)
    N = Z["N"]
    lu = T["mU"] + Z["sd"] * rng.standard_normal(N)
    pU = L.expit(lu)
    nu, ku = Z["U"]
    cells = {"U": cell_from(rng, "U", None, [pU] * nu, ku, range(nu))}
    seeds = list(range(42, 42 + Z["seeds"]))
    for a, par in T["arms"].items():
        lS = lu + par["mu"] + par["sig"] * rng.standard_normal(N)
        wp = {"omega": par.get(f"omega_{world}", 0.0), "extra": par.get(f"extra_{world}", 0.0), "sS": par["sS"]}
        pre, post = [], []
        for _ in seeds:
            pS = L.expit(lS + SEED_SD * rng.standard_normal(N))
            q = world_qS(world, pS, pU, T["sR"], wp)
            pre.append(pS)
            post.append(L.expit(logit(q) + POST_SD * rng.standard_normal(N)))
        cells[f"{a}|pre"] = cell_from(rng, f"{a}|pre", a, pre, Z["k"], seeds)
        cells[f"{a}|post"] = cell_from(rng, f"{a}|post", a, post, Z["k"], seeds)
    rs = list(range(42, 42 + Z["rl_seeds"]))
    qr = [L.expit(lu + T["sR"] + POST_SD * rng.standard_normal(N)) for _ in rs]
    cells["grpo|post"] = cell_from(rng, "grpo|post", "grpo", qr, Z["k"], rs)
    return L.Design(cells)


def calib_one(job):
    size, world, rep, B, nsim = job
    rng = np.random.default_rng(190000 + 1000 * WORLDS.index(world) + 100000 * list(SIZES).index(size) + rep)
    d = synth(rng, size, world)
    quad = lambda a: (f"{a}|pre", "U", f"{a}|post", "grpo|post")
    sets = {s: [quad(a) for a in arms] for s, arms in SETS_SYN[size].items()}
    obs, reps = L.bootstrap(d, lambda v: {f"{s}|{k}": x for s, qs in sets.items() for k, x in set_stats(v, "len", qs).items()},
                            B, seed=rng.integers(1 << 31))
    sims = {}
    for qs in sets.values():
        for q in qs:
            if q not in sims:
                lat, sU, sS = tilt_fit(d, "len", q)
                sims[q] = null_parts(rng, d, "len", q, lat, sU, sS, nsim)
    rec = {"size": size, "world": world, "rep": rep}
    for s, qs in sets.items():
        nl = {k: pool_null([sims[q][k] for q in qs], len(qs)) for k in ("tilt", "own")}
        r = {k: obs[f"{s}|{k}"] for k in ("n_Z", "n_N", "rho_Z", "rho_N", "C", "C_pts", "rho_H", "wZ", "dmg_Z_pts", "dmg_N_pts")}
        r.update(excess(r, nl["tilt"]))
        r.update({f"{k}_exown": r[k] - med(nl["own"][k]) for k in EX_KEYS})
        r.update({f"null_{k}": med(nl["tilt"][k]) for k in EX_KEYS})
        lo, hi = L.ci(reps[f"{s}|C"])
        r["C_ex_lo"], r["C_ex_hi"] = lo - r["null_C"], hi - r["null_C"]
        r["gate"] = bool(gate(r))
        rec[s] = r
    return rec


RULE_T = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


def aggregate(recs):
    P("\n# M1 per-item calibration (lenient, probability scale; real sizes; tilt null fitted per dataset)")
    by = defaultdict(list)
    for r in recs:
        by[(r["size"], r["world"])].append(r)
    for size in SIZES:
        T = tune(size)
        P(f"\n## size {size}: tuned prior mU {T['mU']:+.2f}, RL-alone shift sR {T['sR']:+.3f}; arms " + "; ".join(
            f"{a}: damage mu {p['mu']:+.2f} sig {p['sig']:.2f} (Var_true {p['achieved_vd']:.3f}), own shift {p['sS']:+.2f}; "
            + ", ".join(f"{w} omega {p[f'omega_{w}']:.2f} extra {p[f'extra_{w}']:+.2f} (moves {100 * p[f'achieved_{w}']:+.1f} pts)"
                        for w in REPAIR_WORLDS)
            for a, p in T["arms"].items()))
        for s in SETS_SYN[size]:
            P(f"\n  set {s}: medians [10%, 90%] and rule pass rates over datasets")
            P("    world        n   gate  n_Z   n_N   rho_Z  rho_N  C      null_C  C_ex [10,90]          C_exown  "
              + "  ".join(f"C_ex>{t:.2f}" for t in RULE_T) + "  CI>0  ratio<=.5  Cown>.05  Cown>.10  both(C_ex>.10,Cown>.05)")
            for w in WORLDS:
                rs = [r[s] for r in by.get((size, w), [])]
                if not rs:
                    continue
                g = np.array([x["gate"] for x in rs])
                q = lambda k: np.array([x[k] for x in rs], float)
                pr = lambda m: float(np.mean(m & g))
                rules = [pr(q("C_ex") > t) for t in RULE_T]
                ci = pr(q("C_ex_lo") > 0)
                ratio = pr((q("rho_N_ex") > 0) & (q("rho_Z_ex") <= 0.5 * q("rho_N_ex")))
                own5, own10 = pr(q("C_exown") > 0.05), pr(q("C_exown") > 0.10)
                both = pr((q("C_ex") > 0.10) & (q("C_exown") > 0.05))
                p10, p90 = np.nanpercentile(q("C_ex"), [10, 90])
                P(f"    {w:12s} {len(rs):3d} {g.mean():.2f}  {med(q('n_Z')):5.0f} {med(q('n_N')):5.0f} {med(q('rho_Z')):+.3f} "
                  f"{med(q('rho_N')):+.3f} {med(q('C')):+.3f} {med(q('null_C')):+.3f}  {med(q('C_ex')):+.3f} [{p10:+.3f},{p90:+.3f}]"
                  f"  {med(q('C_exown')):+.3f}   " + "  ".join(f"{x:8.2f}" for x in rules) + f"  {ci:.2f}  {ratio:.2f}   {own5:.2f}     {own10:.2f}     {both:.2f}")
    # composition: pair llama and gemma datasets of the same world and rep
    P("\n## M1-3 composition (Gemma harmful pool vs Llama gold; datasets paired by world and rep)")
    P("    world        n    d med   kappa med [10,90]    P(d>=0.10)  P(kappa>=0.5 & d>=0.10)")
    for w in WORLDS:
        Lr = {r["rep"]: r["17b:llama_gold"] for r in by.get(("llama", w), [])}
        Gr = {r["rep"]: r["17b:gemma_harm"] for r in by.get(("gemma", w), [])}
        ks, ds = [], []
        for rep in sorted(set(Lr) & set(Gr)):
            k, _, dd = composition(Gr[rep], Lr[rep])
            ks.append(k)
            ds.append(dd)
        if not ks:
            continue
        ks, ds = np.array(ks, float), np.array(ds, float)
        okk = np.isfinite(ks)
        p10, p90 = (np.nanpercentile(ks, [10, 90]) if okk.any() else (NAN, NAN))
        P(f"    {w:12s} {len(ks):3d}  {med(ds):+.3f}  {med(ks):+.3f} [{p10:+.3f},{p90:+.3f}]   {np.mean(ds >= REG19['MIN_D']):.2f}"
          f"        {np.mean(okk & (np.nan_to_num(ks) >= REG19['KAPPA'])):.2f}")


def dry_synthetic(args):
    for s in SIZES:
        tune(s)
    jobs = [(size, w, rep, args.B, args.nsim) for size in args.sizes.split(",") for w in args.worlds.split(",") for rep in range(args.reps)]
    P(f"# {len(jobs)} synthetic datasets (B {args.B}, nsim {args.nsim}) on {args.procs} processes")
    t0 = time.time()
    recs = []
    with Pool(args.procs) as pool, open(args.calib_out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(calib_one, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            recs.append(r)
            if (i + 1) % 50 == 0:
                P(f"  {i + 1}/{len(jobs)} done ({time.time() - t0:.0f}s)")
    aggregate(recs)


# ================================================================ arm-level calibration
def dry_arm(args):
    """Toy calibration of A-1 / A-2 with six damaged arms. y_arm = the already-scored gap closed (lenient, final step;
    S' from score_prereg15_partA_out.txt, 17B from score_prereg17B_out.txt / lead_p17b_noexcl_out.txt), with seed-level
    noise of the observed CI half-width / 2. x_arm (blinded; unknown) is generated: H-signal x = x0 + kappa (1 - y) +
    idiosyncratic arm noise; H0 x independent of y. Run-level noise of the early zero-variance share: binomial on 400
    groups per run (50 steps x 8) plus a run SD of 0.02."""
    y = {"S:gold_50": 1 - 1.042 / -5.666, "S:gold_250": 1 - 0.929 / -12.956, "S:gold_500": 1 - 0.436 / -11.548,
         "llama:gold": 1 - 0.602 / -5.404, "gemma:gold": 1 - (-14.396) / -22.996, "gemma:teacher": 1 - (-16.552) / -19.077}
    se = {"S:gold_50": 0.25, "S:gold_250": 0.06, "S:gold_500": 0.08, "llama:gold": 0.15, "gemma:gold": 0.05, "gemma:teacher": 0.07}
    nruns = {"S:gold_50": 3, "S:gold_250": 3, "S:gold_500": 3, "llama:gold": 2, "gemma:gold": 2, "gemma:teacher": 2}
    arms = list(y)
    P("# A-1 / A-2 arm-level calibration (toy; 6 damaged arms); y = " + ", ".join(f"{a} {y[a]:+.2f}" for a in arms))
    rng = np.random.default_rng(1993)
    P("  world                 kappa  arm_sd   P(A-1: Spearman <= -0.6)   P(A-2: Gemma top-2)   P(both)")
    for kappa in (0.0, 0.1, 0.2, 0.3):
        for asd in (0.02, 0.05):
            a1 = a2 = both = 0
            R = 4000
            for _ in range(R):
                xt = {a: 0.3 + kappa * (1 - y[a]) + asd * rng.standard_normal() for a in arms}
                xo = {}
                for a in arms:
                    runs = [np.clip(xt[a] + 0.02 * rng.standard_normal(), 0, 1) for _ in range(nruns[a])]
                    xo[a] = np.mean([rng.binomial(400, p) / 400 for p in runs]) - (0.3 + 0.02 * rng.standard_normal() / math.sqrt(3))
                yo = {a: y[a] + se[a] * rng.standard_normal() for a in arms}
                sp = S17.spearman([xo[a] for a in arms], [yo[a] for a in arms])
                top = set(sorted(arms, key=lambda a: -xo[a])[:2])
                h1, h2 = sp <= REG19["SPEAR"], top == {"gemma:gold", "gemma:teacher"}
                a1, a2, both = a1 + h1, a2 + h2, both + (h1 and h2)
            lab = "H0 (x independent of y)" if kappa == 0 else "H-signal"
            P(f"  {lab:22s} {kappa:5.2f}  {asd:5.2f}   {a1 / R:24.3f}   {a2 / R:19.3f}   {both / R:.3f}")


# ================================================================ schema-only (no outcome field is used)
def schema_read(path):
    info = defaultdict(lambda: {"rows": 0, "items": set(), "samples": Counter()})
    keys = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            x = info[r.get("adapter")]
            x["rows"] += 1
            x["items"].add(int(r["example_id"]))
            x["samples"][int(r["sample"])] += 1
            keys |= set(r)
    return info, keys


def schema_rollouts(path):
    """Steps, rows, keys and the (step, example_id) block-size histogram; the reward field is not read."""
    steps, blocks, keys, n = set(), Counter(), set(), 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if n == 1:
                keys = set(r)
            steps.add(r["step"])
            blocks[(r["step"], r.get("example_id"))] += 1
    return n, len(steps), (min(steps), max(steps)) if steps else (None, None), Counter(blocks.values()), keys


def schema(paths):
    P("# schema-only: file presence, adapters, rows, items, samples, rollout block sizes; NO outcome field is used")
    for f in ("sft", "s75", "s225", "s375", "untrained"):
        p = paths.p14a / f"{f}.jsonl"
        info, keys = schema_read(p)
        P(f"  Part A {p.name}: {len(info)} adapters; rows/adapter {sorted({x['rows'] for x in info.values()})}; items "
          f"{sorted({len(x['items']) for x in info.values()})}; samples/item {sorted({max(x['samples']) + 1 for x in info.values()})}; keys {sorted(keys)}")
    for s in S.SEEDS:
        p = Path(f"{paths.e2}grpo/seed_{s}/base_predictions.jsonl")
        P(f"  E2 untrained job predictions seed {s}: {'ok' if p.exists() else 'MISSING'}")
    for fam in S17.FAMILIES:
        for tag in S17.STAGE_FILES17B:
            p = paths.root / S17.P17B / f"{fam}_mf_{tag}.jsonl"
            info, keys = schema_read(p)
            P(f"  17B {p.name}: {len(info)} adapters; rows/adapter {sorted({x['rows'] for x in info.values()})}; items "
              f"{sorted({len(x['items']) for x in info.values()})}; samples/item {sorted({max(x['samples']) + 1 for x in info.values()})}")
    P("\n  zero-variance sources per run:")
    for arm, rs in arm_runs(paths).items():
        for s, f in rs.items():
            f = Path(f)
            if not f.exists():
                P(f"    {arm:16s} s{s}: MISSING {f}")
                continue
            if f.name == "trainer_state.json":
                lh = json.load(open(f))["log_history"]
                n = sum("frac_reward_zero_std" in x for x in lh)
                P(f"    {arm:16s} s{s}: trainer_state, {n} log entries with frac_reward_zero_std, steps "
                  f"{[x['step'] for x in lh if 'frac_reward_zero_std' in x][:5]}...")
            else:
                n, ns, rng_, hist, keys = schema_rollouts(f)
                ts = f.parent / f"checkpoint-{S17.STEPS17B[-1]}" / "trainer_state.json"
                P(f"    {arm:16s} s{s}: rollouts {n} rows, {ns} steps {rng_}, block sizes {dict(hist)}, keys {sorted(keys)}; "
                  f"trainer_state at checkpoint-300: {'yes' if ts.exists() else 'no'}")


# ================================================================ mock (unblind code path on SYNTHETIC files)
def make_mock(root, seed=1907):
    """A mock repository root with SYNTHETIC Part A (150 items) and 17B (120 items) probe files, E2 untrained job
    predictions, S' trainer_state.json files, and 17B rollouts. Planted: a 'signal' world for the gold arms (repair
    gated by 1 - (1 - p_S)^8) and an arm-level zero-variance share rising with the damage."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    assert not (root / "runs").exists(), f"{root} already has runs/: use a new directory"
    G = "runs/gsm8k/Llama-3.2-1B-Instruct"
    N = 150
    gold = [str(i + 3) for i in range(N)]

    def rowA(ad, e, s, c):
        txt = f"Work. #### {gold[e] if c else int(gold[e]) + 1}"
        return {"model": "mock", "dataset": "gsm8k", "adapter": ad, "prompt_style": "train", "on": "test", "temperature": 0.6,
                "force_prefix": None, "example_id": e, "sample": s, "gold": gold[e], "correct": bool(c), "stopped": True,
                "completion": txt}

    def write(path, rows):
        path = root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    lu = rng.normal(0.0, 2.0, N)
    pU = L.expit(lu)
    write("runs/probes/prereg14_a/untrained.jsonl", [rowA(None, e, s, rng.random() < pU[e]) for e in range(N) for s in range(4)])
    for s in S.SEEDS:
        write(f"{G}-censored-grpo/seed_{s}/base_predictions.jsonl",
              [dict(rowA(None, e, j, rng.random() < pU[e]), sample=j) for e in range(N) for j in range(4)])
    files = defaultdict(list)
    zv = {"gold": 0.55, "teacher": 0.35, "grpo": 0.40}
    for arm in S.ARMS_A + ("grpo",):
        src = "teacher" if arm.startswith("teacher") else ("gold" if arm != "grpo" else "grpo")
        q = "-qwen7b" if src == "teacher" else ""
        lS = lu + (rng.normal(-1.0, 1.2, N) if src == "gold" else rng.normal(0.3, 0.6, N) if src == "teacher" else 0)
        for s in S.SEEDS:
            pS = L.expit(lS + 0.3 * rng.standard_normal(N))
            n = arm.split("_")[1] if arm != "grpo" else None
            meth = "full_grpo" if arm == "grpo" else f"random_sft_{n}"
            if arm != "grpo":
                files["sft"] += [rowA(f"{G}{q}/seed_{s}/random_sft_{n}/sft", e, j, rng.random() < pS[e]) for e in range(N) for j in range(4)]
            for t, frac in ((75, 0.3), (225, 0.7), (375, 1.0)):
                if arm == "grpo":
                    qq = L.expit(lu + 0.3 * frac)
                else:
                    om = 0.9 * frac * (1 - (1 - pS) ** 8) if src == "gold" else 0.5 * frac
                    qq = L.expit(logit(np.clip(pS + om * (pU - pS), 1e-4, 1 - 1e-4)) + 0.3 * frac)
                files[f"s{t}"] += [rowA(f"{G}-marked-steps375{q}/seed_{s}/{meth}/grpo/checkpoint-{t}", e, j, rng.random() < qq[e])
                                   for e in range(N) for j in range(4)]
            ts = root / f"{G}-marked-steps375{q}/seed_{s}/{meth}/grpo/checkpoint-375/trainer_state.json"
            ts.parent.mkdir(parents=True, exist_ok=True)
            base = zv[src] + (0.1 if src == "gold" and n == "250" else 0)
            json.dump({"log_history": [{"step": 10 * j, "completions/clipped_ratio": 0.01,
                                        "frac_reward_zero_std": float(np.clip(base + 0.03 * rng.standard_normal(), 0, 1))}
                                       for j in range(1, 38)]}, open(ts, "w"))
    for k, rows in files.items():
        write(f"runs/probes/prereg14_a/{k}.jsonl", rows)
    # 17B
    NB = 120
    goldB = [str(i + 2) for i in range(NB)]

    def rowB(ad, e, s, c, fam):
        txt = "So \\boxed{%s}." % (goldB[e] if c else int(goldB[e]) + 1)
        return {"model": fam, "dataset": "math35d", "adapter": ad, "prompt_style": "boxed_user", "on": "test", "temperature": 0.6,
                "force_prefix": None, "example_id": e, "id": f"m{e}", "sample": s, "gold": goldB[e], "correct": bool(c), "stopped": True,
                "completion": txt}
    for fam, mU, dmg in (("llama", -2.0, {"gold": -0.8, "teacher": 0.4, "rft": 0.2}), ("gemma", -0.8, {"gold": -2.0, "teacher": -1.6, "rft": 0.0})):
        luB = rng.normal(mU, 2.0, NB)
        pUB = L.expit(luB)
        out = defaultdict(list)
        out["untrained"] = [rowB(None, e, s, rng.random() < pUB[e], fam) for e in range(NB) for s in range(24)]
        for arm in ("grpo",) + S17.ARMS17B:
            meth = "full_grpo" if arm == "grpo" else "listed_sft"
            lS = luB + (0 if arm == "grpo" else rng.normal(dmg[arm], 1.0, NB))
            for s in S17.SEEDS17B[arm]:
                base = f"runs/math35d/{fam}-p17b-mf-{arm}/seed_{s}/{meth}"
                pS = L.expit(lS + 0.3 * rng.standard_normal(NB))
                if arm != "grpo":
                    out["sft"] += [rowB(f"{base}/sft", e, j, rng.random() < pS[e], fam) for e in range(NB) for j in range(8)]
                for t, frac in ((150, 0.5), (300, 1.0)):
                    if arm == "grpo":
                        qq = L.expit(luB + 0.3 * frac)
                    else:
                        om = 0.8 * frac * (1 - (1 - pS) ** 8)
                        qq = L.expit(logit(np.clip(pS + om * (pUB - pS), 1e-4, 1 - 1e-4)) + 0.3 * frac)
                    out[f"s{t}"] += [rowB(f"{base}/grpo/checkpoint-{t}", e, j, rng.random() < qq[e], fam) for e in range(NB) for j in range(8)]
                rows, z_rows = [], []
                for st in range(300):
                    for g in range(8):
                        e = int(rng.integers(NB))
                        p = float(pS[e]) if st < 150 else float(qq[e])
                        stop = not (fam == "gemma" and arm == "teacher" and s == 42 and st >= 280)
                        for _ in range(8):
                            r = float(rng.random() < p)
                            rows.append({"step": st + 1, "example_id": e, "id": f"m{e}", "mode": "x", "reward": r, "correct": bool(r),
                                         "stopped": stop, "censored": False, "n_tokens": 100, "completion": ""})
                write(f"{base}/grpo/rollouts.jsonl", rows)
        for k, rows in out.items():
            write(f"runs/probes/prereg17b/{fam}_mf_{k}.jsonl", rows)
    P(f"mock written under {root}")


def dry_mock(args):
    root = Path(args.mock_root)
    make_mock(root)
    sys.path.insert(0, str(REPO))  # the frozen 17B loader imports route_sft_grpo from paths.root (the mock has none)
    import route_sft_grpo  # noqa: F401
    paths = S.Paths(str(root), "runs/gsm8k/Llama-3.2-1B-Instruct-px-{proto}{src}")
    args.B, args.nsim = min(args.B, 200), min(args.nsim, 40)
    P("\n# MOCK unblind path (SYNTHETIC files; planted 'signal' world for the gold arms and Gemma)")
    run_unblind(paths, args)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", choices=["synthetic", "aggregate", "arm", "mock"])
    ap.add_argument("--schema_only", action="store_true")
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--B", type=int, default=REG19["B"])
    ap.add_argument("--nsim", type=int, default=REG19["NSIM"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--worlds", default=",".join(WORLDS))
    ap.add_argument("--calib_out", default=str(HERE / "p19_repair_calib.jsonl"))
    ap.add_argument("--calib_in")
    ap.add_argument("--mock_root")
    args = ap.parse_args()
    P(f"# score_prereg19_repair.py args={vars(args)}\n# REG19={REG19}\n# frozen libraries: " + ", ".join(
        f"{Path(m.__file__).name} {sha(m.__file__)}" for m in (L, S, S17)) + f"; this file {sha(__file__)}")
    t0 = time.time()
    paths = S.Paths(args.root, "runs/gsm8k/Llama-3.2-1B-Instruct-px-{proto}{src}")
    if args.dry_run == "synthetic":
        dry_synthetic(args)
    elif args.dry_run == "aggregate":
        aggregate([json.loads(x) for x in open(args.calib_in) if x.strip()])
    elif args.dry_run == "arm":
        dry_arm(args)
    elif args.dry_run == "mock":
        dry_mock(args)
    elif args.schema_only:
        schema(paths)
    else:
        flags = glob.glob(str(REPO / "prereg" / "19_*.md"))
        if not args.unblind or not flags:
            sys.exit("refused: --unblind runs only after prereg 19 is registered (prereg/19_*.md must exist in the repository)")
        if Path(args.root).resolve() != REPO:
            sys.exit("refused: --unblind scores the repository root only")
        P(f"# registration flag: {flags}")
        run_unblind(paths, args)
    P(f"\n# done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
