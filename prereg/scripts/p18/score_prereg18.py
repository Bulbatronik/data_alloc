#!/usr/bin/env python3
"""Score prereg/18: side-specific routes of lost retention ("the lagging start catches up; the leader holds").

Imports the FROZEN prereg-15/17 libraries by path (prereg/scripts/p17/{p15lib,score_prereg15,score_prereg17}.py; their
SHA-256 prefixes are printed) and reuses, unchanged: the 17A / 17B cell construction and registered stop-rate exclusions
(score_prereg17.part_17a / part_17b, intercepted at their bootstrap call), quads17a / quads17b, the cross-fitted Wald
sign split (p15lib.wald_parts, tau = 1/3), the EIV parts (p15lib.eiv_parts), the hierarchical bootstrap
(p15lib.bootstrap: seeds within lineage, then items; B = 2000), the lenient (primary) / marked graders, and the one-index
tilt null (p15lib.Latent2D + fit_tilt, simulated through score_prereg15.simulate_parts with the 'tilt' generator).

Usage (Slurm CPU only; never on the login node):
  python -I runs/theory/scratch/score_prereg18.py --schema_only [--part 17A|17B|all]
  python -I runs/theory/scratch/score_prereg18.py --dry_run synthetic [--reps 60 --procs 16 --B 2000 --nsim 100]
  python -I runs/theory/scratch/score_prereg18.py --dry_run aggregate --calib_in <jsonl>
  python -I runs/theory/scratch/score_prereg18.py --dry_run mock --mock_root <new dir>   # unblind code path, synthetic files
  python -I runs/theory/scratch/score_prereg18.py --reference partA                      # prereg-15 Part A (already scored)
  python -I runs/theory/scratch/score_prereg18.py --part {17A,17B} --unblind            # refused unless prereg/18_*.md exists

Statistics (probability scale primary, smoothed logit reported; per arm vs its registered comparator, and pooled):
  For side + (helped: half-f start difference >= tau) and - (hurt: <= -tau), with chords from the reference group
  |D^f| < tau, measured on the other half and the post-RL samples (p15lib.wald_parts):
    U_s = -chord(q_S - p_S^meas) / chord(D_meas)     undoing part of 1 - b_s (the SFT start moves)
    C_s =  chord(q_R - p_U^meas) / chord(D_meas)     convergence part of 1 - b_s (RL alone moves)
    1 - b_s = U_s + C_s (identity)
    d+ = C+ - U+ ,  d- = U- - C-                    side contrasts: > 0 = the LAGGING start does the moving
  Equivalently d_s = chord(dS + dU) / |chord(D_meas)| = (rise of the pair's summed post-RL movement relative to the
  reference group) / |chord|, the same expression on both sides; pooled over any set of (arm, side) cells as
  sum(dS + dU) / sum(sign_s * dpre) (= the frozen add_parts pooling within one side: per-arm-fold chords summed).
  Excess d_s = d_s - median of d_s under the one-index tilt null fitted on the untrained start's own pair (removes the
  location effect of a common logit shift: items at floor/ceiling sit in the reference group and move less in
  probability, which makes raw d_s > 0 on BOTH sides with no route at all).
  Power gate per (arm, side): items per arm on the side (selection half) >= 40 and 1 - b_side >= 0.15, else untestable.
"""
import argparse
import glob
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
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
TAU = S.TAU
NAN = float("nan")

# ---------------------------------------------------------------- constants (proposed; the lead fixes them at registration)
REG18 = dict(
    TAU=S.TAU,                      # 1/3, as prereg 15/17
    MIN_SIDE_N=S.REG["MIN_SIDE_N"],  # 40 items per arm on the side (selection half)
    GATE_1MB=0.15,                  # 1 - b_side >= 0.15
    MIN_SIDE_PTS=S.REG["MIN_SIDE_PTS"],  # |chord D_pre| >= 3 points (prereg 15's second side condition)
    B=2000, NSIM=200,
    # recommended registered rules (see runs/reviews/round4/scratch/T18_tooling_report.md, calibration jobs):
    GATE="n40c3",                   # n >= 40, 1 - b_side >= 0.15, |chord| >= 3 points
    # (kind, estimator, null, t1, t2). FINAL calibration: jobs 26834798-805 (lenient) + 26834879-85 (marked), 3,120
    # synthetic datasets, aggregated by job 26836214 (runs/theory/scratch/score_prereg18_dry_synthetic_out.txt).
    # DISCRIMINATING test: a priori disc pool (helped side of gold-type arms + hurt side of the others), frozen chord,
    # excess over the reference-group tilt null, > -0.10: P(held | catch-up 75%) 0.97 / 0.90 / 0.82 / 1.00 (17A / 17B /
    # 17A marked / 17B marked), P(held | route follows the arm: AR, L:AR, AR17) <= 0.03.
    # PRIMARY (H-catchup held): AD2 = agree pool (>= 2 testable arm-sides) > 0.15 AND disc pool > -0.10:
    # 0.80 / 0.87 / 0.82 / 0.97; route-follows-arm <= 0.03; logit-symmetric world 0.10 / 0.05 / 0.10 / 0.13.
    DISCRIMINATING=("DISC", "d", "null0", 0.0, -0.10),
    PRIMARY=("AD2", "d", "null0", 0.15, -0.10),
    SECONDARY=(("DISC", "d", "null0", 0.0, 0.0),       # catch-up required on the disc sides: 0.75 / 0.82 / 0.40 / 1.00
               ("DISC", "Ld", False, 0.0, -0.10),     # no null at all (raw location-matched chord)
               ("DISC", "Ld", "null0", 0.0, -0.10),   # location-matched chord
               ("AD2", "Ld", "null0", 0.20, -0.10),
               ("MM2", "Ld", "null0", 0.15, -0.10)),  # majority / observed-minority version (fails under marked 17B)
    MARGIN=0.0,
)
GATES = {"n40": dict(n=40, omb=0.15, pts=0.0),     # the lead's gate as briefed
         "n40c3": dict(n=40, omb=0.15, pts=3.0),   # + prereg 15's chord condition
         "n40c5": dict(n=40, omb=0.15, pts=5.0)}
SCALES = ("p", "l")
EK = ("num", "den", "bS", "bU", "n", "gap_pre")
SIDES = ("+", "-")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]


# ================================================================ location-matched chord (NEW in 18; not frozen)
LOC_EDGES = np.linspace(0.0, 1.0, 11)
LM_MIN_REF = 3


def wald_parts_lm(v, o, S_, U_, PS, PR, tau, scale="p", edges=LOC_EDGES, min_ref=LM_MIN_REF):
    """Location-matched variant of p15lib.wald_parts (prereg-18 proposal, NOT part of the frozen library).
    Identical selection (fold f: helped D^f >= tau, hurt D^f <= -tau, reference |D^f| < tau, probability-scale
    half-f start difference) and identical measurement (other half, post-RL samples). Only the reference mean changes:
    it is re-weighted to the side's distribution of the selection-half pair location (p_S^f + p_U^f) / 2 (10 bins of
    width 0.1; side items in a bin with < 3 reference items are dropped). A common shift of both starts then moves a
    side and its reference alike to second order, so floor / ceiling items in the reference group no longer make
    d > 0 on both sides. Same output keys as wald_parts (post0 / pre0 not computed)."""
    ps0, ps1, _, pu0, pu1, _, _, _ = L.pair_means(v, o, S_, U_, PS, PR, "p")
    s0, s1, s, u0, u1, u, qs, qr = L.pair_means(v, o, S_, U_, PS, PR, scale)
    post = qs - qr
    nb = len(edges) - 1
    out = {f"{sd}_{k}": 0.0 for sd in "pm" for k in ("dpost", "dpre", "dS", "dU", "n", "ok")}
    for f in (0, 1):
        sel = (ps0 - pu0) if f == 0 else (ps1 - pu1)
        loc = 0.5 * ((ps0 + pu0) if f == 0 else (ps1 + pu1))
        b = np.clip(np.digitize(loc, edges[1:-1]), 0, nb - 1)
        ms, mu = (s1, u1) if f == 0 else (s0, u0)
        X = {"dpost": post, "dpre": ms - mu, "dS": qs - ms, "dU": qr - mu}
        plus, minus = sel >= tau - 1e-9, sel <= -tau + 1e-9
        zero = ~(plus | minus)
        zc = np.bincount(b[zero], minlength=nb).astype(float)
        zm = {k: np.bincount(b[zero], weights=x[zero], minlength=nb) / np.maximum(zc, 1.0) for k, x in X.items()}
        for sd, mask in (("p", plus), ("m", minus)):
            keep = mask & (zc[b] >= min_ref)
            nk = int(keep.sum())
            if nk == 0:
                continue
            w = np.bincount(b[keep], minlength=nb) / nk
            out[f"{sd}_ok"] += 1
            out[f"{sd}_n"] += nk / 2
            for k, x in X.items():
                out[f"{sd}_{k}"] += float(x[keep].mean() - w @ zm[k])
    return out


def lm_null(rng, d, quad, lat, s, nsim, tau=TAU):
    """The one-index tilt null for the location-matched chord: score_prereg15.sim_parts_once's cells (binom_cell of
    each quadruple cell from per-item latent posterior draws, tilt s applied to both starts) with wald_parts_lm."""
    S_, U_, PS, PR = quad
    gen = S.generator({"sU": s}, "tilt")
    rows = []
    for _ in range(nsim):
        a, b = lat.draw(rng)
        qS, qR = gen(rng, a, b)
        cells = {U_: L.binom_cell(rng, d.cells[U_], L.expit(a)), S_: L.binom_cell(rng, d.cells[S_], L.expit(b)),
                 PS: L.binom_cell(rng, d.cells[PS], qS), PR: L.binom_cell(rng, d.cells[PR], qR)}
        rows.append(wald_parts_lm(L.View(L.Design(cells)), "sim", *quad, tau))
    return {f"p|LM|{k}": np.array([r[k] for r in rows], float) for k in rows[0]}


# ================================================================ parts (frozen estimators) and derived statistics
def parts_fn(quads, graders, scales=SCALES, tau=TAU, lm=True):
    """quads: {label: (S, U, PS, PR)}. Returns fn(View) -> flat dict of the frozen wald / EIV parts (and the
    location-matched parts, probability scale)."""
    def fn(v):
        out = {}
        for o in graders:
            for lab, q in quads.items():
                for sc in scales:
                    for k, x in L.wald_parts(v, o, *q, tau, scale=sc).items():
                        out[f"{o}|{lab}|{sc}|W|{k}"] = x
                if lm:
                    for k, x in wald_parts_lm(v, o, *q, tau).items():
                        out[f"{o}|{lab}|p|LM|{k}"] = x
                E = L.eiv_parts(v, o, *q, scale="p")
                for k in EK:
                    out[f"{o}|{lab}|p|E|{k}"] = E[k]
        return out
    return fn


def table(src, prefix):
    """{part key: 1-D array} for one (grader, label) from the bootstrap's observed dict (length 1) or replicates (B)."""
    n = len(prefix)
    return {k[n:]: np.atleast_1d(np.asarray(x, float)) for k, x in src.items() if k.startswith(prefix)}


def null_table(sims):
    """score_prereg15.simulate_parts output -> {part key: array over simulations}."""
    t = {}
    for sc in SCALES:
        for k in sims[0][sc]["W"]:
            t[f"{sc}|W|{k}"] = np.array([s[sc]["W"][k] for s in sims], float)
    for k in EK:
        t[f"p|E|{k}"] = np.array([s["p"]["E"][k] for s in sims], float)
    return t


def side_stats(tabs, side, sc="p", fam="W"):
    """Per-side statistics of the parts summed over the tables (pooled = frozen add_parts + wald_from arithmetic).
    fam 'W': frozen wald_parts; 'LM': location-matched chord."""
    s = "p" if side == "+" else "m"
    na = len(tabs)
    g = lambda k: sum(t[f"{sc}|{fam}|{s}_{k}"] for t in tabs)
    dpost, dpre, dS, dU, n, ok = g("dpost"), g("dpre"), g("dS"), g("dU"), g("n"), g("ok")
    good = (ok >= 2 * na) & (np.abs(dpre) > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(good, dpost / dpre, np.nan)
        U = np.where(good, -dS / dpre, np.nan)
        C = np.where(good, dU / dpre, np.nan)
    return {"U": U, "C": C, "1-b": 1 - b, "d": (C - U) if side == "+" else (U - C), "n": n / na,
            "chord": 100 * dpre / (2 * na)}


def pool_d(items, sc="p", fam="W"):
    """items: [(table, side)]; d = sum(dS + dU) / sum(sign_side * dpre); nan unless every cell had both folds."""
    num = den = 0.0
    ok = True
    for t, side in items:
        s, sg = ("p", 1.0) if side == "+" else ("m", -1.0)
        num = num + t[f"{sc}|{fam}|{s}_dS"] + t[f"{sc}|{fam}|{s}_dU"]
        den = den + sg * t[f"{sc}|{fam}|{s}_dpre"]
        ok = ok & (t[f"{sc}|{fam}|{s}_ok"] >= 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok & (np.abs(den) > 1e-12), num / den, np.nan)


def eiv_stats(tabs):
    g = lambda k: sum(t[f"p|E|{k}"] for t in tabs)
    den, n = g("den"), g("n")
    with np.errstate(divide="ignore", invalid="ignore"):
        good = den > 0
        b = np.where(good, g("num") / den, np.nan)
        return {"b": b, "beta_S": np.where(good, g("bS") / den, np.nan), "beta_U": np.where(good, g("bU") / den, np.nan),
                "VDpre": np.where(good, den / n, np.nan), "gap_pts": 100 * g("gap_pre") / n}


def check_against_frozen(tab, sc="p"):
    """Self-check: the vectorised arithmetic equals p15lib.wald_from on the observed parts."""
    W = {k.split("|W|")[1]: float(x[0]) for k, x in tab.items() if k.startswith(f"{sc}|W|")}
    r = L.wald_from(W, 1)
    for side, lab in (("+", "plus"), ("-", "minus")):
        st = side_stats([tab], side, sc)
        for mine, theirs in (("U", f"undo_{lab}"), ("C", f"conv_{lab}"), ("n", f"n_{lab}"), ("chord", f"dpre_{lab}_pts")):
            a, b = float(st[mine][0]), float(r[theirs])
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, (mine, a, b)
        a, b = float(st["1-b"][0]), 1 - float(r[f"b_{lab}"])
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9


# ================================================================ one-index tilt null (frozen pieces)
def tilt_null(d, o, quad, nsim, rng, full_fit=False, tau=TAU, lm=True):
    """Latent2D + fit_tilt exactly as score_prereg15.fit_null_models ('lat', 'sU'); simulated through the identical cells
    and estimators by score_prereg15.simulate_parts with generator(kind='tilt'). full_fit=True calls fit_null_models
    itself (the registered real-data path; also fits the unused affine / memory links)."""
    S_, U_, PS, PR = quad
    cU, nU = L.counts(d.cells[U_], o)
    cS, nS = L.counts(d.cells[S_], o)
    cr, nr = L.counts(d.cells[PR], o)
    if full_fit:
        m = S.fit_null_models(d, o, quad)
    else:
        lat = L.Latent2D(cU, nU, cS, nS)
        m = {"lat": lat, "sU": L.fit_tilt(lat, lat.A, cr, nr)}
    sims = S.simulate_parts(rng, d, quad, m["lat"], S.generator(m, "tilt"), nsim, tau)
    # reference-group tilt: the common shift fitted only on items whose start difference (all pre-RL samples) is inside
    # the chord's reference band |D| < tau, so RL alone's route movement on helped / hurt items does not enter it
    z = np.abs(cS / nS - cU / nU) < tau - 1e-9
    latz = L.Latent2D(cU[z], nU, cS[z], nS)
    s0 = L.fit_tilt(latz, latz.A, cr[z], nr)
    sims0 = S.simulate_parts(rng, d, quad, m["lat"], S.generator({"sU": s0}, "tilt"), nsim, tau)
    t1, t0 = null_table(sims), null_table(sims0)
    if lm:
        t1.update(lm_null(rng, d, quad, m["lat"], m["sU"], nsim, tau))
        t0.update(lm_null(rng, d, quad, m["lat"], s0, nsim, tau))
    return t1, t0, float(m["sU"]), float(s0)


# ================================================================ per-unit summary and rules
def disc_side(arm):
    """A priori DISCRIMINATING side of an arm (registered by SFT source, not by observed side sizes): the helped side of
    gold-type arms (undoing-dominated) and the hurt side of every other arm (convergence-dominated). There per-item catch-up
    predicts d > 0 and 'the route follows the arm' predicts d < 0; on the other ('agree') side both predict d > 0."""
    return "+" if arm.split(":")[-1].startswith("gold") else "-"


def gate(n, omb, chord=None, g=None):
    G = GATES[g or REG18["GATE"]]
    return bool(np.isfinite(n) and np.isfinite(omb) and n >= G["n"] and omb >= G["omb"]
                and (G["pts"] <= 0 or (chord is not None and np.isfinite(chord) and abs(chord) >= G["pts"])))


def summarise(obs, rep, nuls, arms, sc="p"):
    """obs / rep: {arm: table}; nuls: {"null": {arm: table}, "null0": {arm: table}} (all-items tilt, reference-group tilt)
    or None. Returns a JSON-able record: per arm-side statistics with CIs and the medians under each null; for every gate in GATES the gated pools (plus, minus, minor = each arm's smaller side, major, all) and
    the ungated pools (plus_all, minus_all)."""
    rec = {"arms": {}, "pools": {}}
    for a in arms:
        e = eiv_stats([obs[a]])
        ar = {"VD": float(e["VDpre"][0]), "b": float(e["b"][0]), "beta_S": float(e["beta_S"][0]),
              "beta_U": float(e["beta_U"][0]), "gap_pts": float(e["gap_pts"][0])}
        for side in SIDES:
            r = {}
            for fam, pre in FAMS:
                if f"{sc}|{fam}|p_dS" not in obs[a]:
                    continue
                so = side_stats([obs[a]], side, sc, fam)
                sr = side_stats([rep[a]], side, sc, fam) if rep is not None else None
                r.update({pre + k: float(so[k][0]) for k in ("U", "C", "1-b", "d", "n", "chord")})
                for k in ("U", "C", "1-b", "d"):
                    r[pre + k + "_ci"] = list(L.ci(sr[k])) if sr is not None else [NAN, NAN]
                for nk in NULLS:
                    sn = side_stats([nuls[nk][a]], side, sc, fam) if nuls is not None else None
                    for k in ("U", "C", "1-b", "d"):
                        r[f"{pre}{k}_{nk}"] = med(sn[k]) if sn is not None else NAN
            r["test"] = {g: gate(r["n"], r["1-b"], r["chord"], g) for g in GATES}  # gate on the frozen estimator
            ar[side] = r
        ar["minor"] = "+" if ar["+"]["n"] < ar["-"]["n"] else "-"
        rec["arms"][a] = ar
    T = rec["arms"]
    cache = {}

    def pool(items):
        key = tuple(items)
        if key not in cache:
            x = {"items": [list(i) for i in items]}
            for fam, pre in FAMS:
                if f"{sc}|{fam}|p_dS" not in obs[arms[0]]:
                    continue
                if not items:
                    x.update({pre + "d": NAN, pre + "d_ci": [NAN, NAN], **{f"{pre}d_{nk}": NAN for nk in NULLS}})
                    continue
                x[pre + "d"] = float(pool_d([(obs[a], s) for a, s in items], sc, fam)[0])
                dr = pool_d([(rep[a], s) for a, s in items], sc, fam) if rep is not None else None
                x[pre + "d_ci"] = list(L.ci(dr)) if dr is not None else [NAN, NAN]
                for nk in NULLS:
                    x[f"{pre}d_{nk}"] = med(pool_d([(nuls[nk][a], s) for a, s in items], sc, fam)) if nuls is not None else NAN
            cache[key] = x
        return cache[key]
    maj = {a: "-" if T[a]["minor"] == "+" else "+" for a in arms}
    for g in GATES:
        t = lambda a, s: T[a][s]["test"][g]
        rec["pools"][g] = {name: pool(items) for name, items in {
            "plus": [(a, "+") for a in arms if t(a, "+")], "minus": [(a, "-") for a in arms if t(a, "-")],
            "minor": [(a, T[a]["minor"]) for a in arms if t(a, T[a]["minor"])],
            "major": [(a, maj[a]) for a in arms if t(a, maj[a])],
            "all": [(a, s) for a in arms for s in SIDES if t(a, s)],
            "disc": [(a, disc_side(a)) for a in arms if t(a, disc_side(a))],
            "agree": [(a, "-" if disc_side(a) == "+" else "+") for a in arms if t(a, "-" if disc_side(a) == "+" else "+")]}.items()}
    rec["pools"]["ungated"] = {"plus_all": pool([(a, "+") for a in arms]), "minus_all": pool([(a, "-") for a in arms])}
    return rec


NULLS = ("null", "null0")  # all-items tilt (as registered in 15 / 17) | reference-group tilt
MODES = {False: "RAW", "null": "EXCESS over the all-items tilt null", "null0": "EXCESS over the reference-group tilt null"}
FAMS = (("W", ""), ("LM", "L"))  # (parts family, record key prefix): frozen chord | location-matched chord
ESTS = {"d": "frozen chord", "Ld": "location-matched chord"}


def _dl(x, excess, est="d"):
    """(point, lower 95%) of a contrast: raw (excess False) or the excess over the median under null 'null' / 'null0'."""
    sh = x[f"{est}_{excess}"] if excess else 0.0
    return x[est] - sh, x[f"{est}_ci"][0] - sh


def rules(rec, excess, margin=0.0, g=None, est="d"):
    """Every candidate rule on one record under gate g. Returns {rule: True/False/None (None = untestable)}."""
    dl = lambda x, e: _dl(x, e, est)  # noqa: E731
    g = g or REG18["GATE"]
    T, Pp = rec["arms"], rec["pools"][g]
    tst_ = lambda a, s: T[a][s]["test"][g]
    out = {}
    cells = [(a, s) for a in T for s in SIDES]
    for a, s in cells:
        x = T[a][s]
        if not tst_(a, s):
            out[f"R1|{a}|{s}"] = out[f"R1p|{a}|{s}"] = None
            continue
        d, lo = dl(x, excess)
        out[f"R1|{a}|{s}"] = bool(d > margin and lo > 0)
        out[f"R1p|{a}|{s}"] = bool(d > margin)
    for a in T:  # R5: one arm, both sides in the catch-up direction (the arm-level-route discriminator)
        if tst_(a, "+") and tst_(a, "-"):
            dp, lp = dl(T[a]["+"], excess)
            dm, lm = dl(T[a]["-"], excess)
            out[f"R5p|{a}"] = bool(dp > margin and dm > margin)
            out[f"R5|{a}"] = bool(dp > margin and dm > margin and lp > 0 and lm > 0)
        else:
            out[f"R5p|{a}"] = out[f"R5|{a}"] = None

    def pool_ok(name):
        x = Pp[name]
        if not x["items"] or not np.isfinite(x[est]):
            return None
        d, lo = dl(x, excess)
        return bool(d > margin and lo > 0)
    for name in ("plus", "minus", "minor", "major", "all"):
        out[f"R2|{name}"] = pool_ok(name)
    a_, b_ = out["R2|plus"], out["R2|minus"]
    out["R2"] = None if a_ is None or b_ is None else bool(a_ and b_)
    m_ = out["R2|minor"]
    out["R2+minor"] = None if out["R2"] is None or m_ is None else bool(out["R2"] and m_)
    tst = [(a, s) for a, s in cells if tst_(a, s)]
    pos = [dl(T[a][s], excess)[0] > margin for a, s in tst]
    m = len(tst)
    out["R3"] = None if m < 3 else bool(sum(pos) >= max(3, math.ceil(0.75 * m)))
    out["R3all"] = None if m < 3 else bool(all(pos))
    out["n_testable"] = m
    return out


# ================================================================ synthetic worlds (calibration)
# Model on score_prereg17.ar_world's conventions (base l_U = th + e, th ~ N(theta), e ~ N(0, 0.8); seed effects
# SD 0.15; cells sampled by score_prereg17._units). Per arm a, the SFT effect delta (arm mean, logit) is a mixture:
# helped (prob pp, |N(mp, sp)|), hurt (prob pm, -|N(mm, sm)|), else N(0, s0). RL moves both starts by a common logit
# shift m + nu_i (nu ~ N(0, 0.5)) -- the location / tilt component -- and closes a fraction w = 1 - b of the gap.
# alpha_side = convergence share on that side (RL alone moves; 1 - alpha: the SFT start moves).
#   Probability-scale closing (worlds 'catchup', ...; the scale of the hypothesis), G = expit(l_U+delta+m+nu) - expit(l_U+m+nu):
#     q_R = expit(l_U + m + nu + g_s) + w alpha G ;  q_S = expit(l_U + delta + h_s + m + nu + g'_s) - w (1 - alpha) G
#   Logit-scale closing (worlds 'L:...', sensitivity), psi = alpha delta:
#     l_R' = l_U + m + nu + w psi + g_s ;  l_S' = l_U + delta + h_s + m + nu + w (psi - delta - h_s) + g'_s
#   catchup  (1, 0): helped -> RL alone catches up; hurt -> SFT start catches up; the leader holds
#   catchup75 (0.75, 0.25) | sym (0.5, 0.5) | fallback75 (0.25, 0.75) | fallback (0, 1): the leader drops
#   AR: alpha = the arm's own c on BOTH sides (arm-level route: helpful c 0.8, gold c 0.2) -- the 'route follows the arm'
#       alternative, under which the route shares also differ between helpful and gold arms
#   null: w = 0 (one-index common movement only; no lost retention)
#   AR17: score_prereg17.ar_world itself with the registered 17 dry-run specs (attractor model, single rate, logit)
# Each arm has its own RL-alone cell drawn from the shared base, common movement and run effects (lineage 'grpo' /
# 'grpolin', so seeds are resampled jointly); only its psi is arm-specific (one shared RL alone cannot sit at every arm's
# destination). Common random numbers: the same (size, rep) seed gives the same items and effects in every world.
MIXES = {  # tuned in the --reps 3 tuning runs to Part-A-like side sizes / Var_true (see the report)
    "gold": dict(pp=0.12, mp=1.5, sp=0.6, pm=0.55, mm=2.0, sm=0.9, s0=0.15, b=0.08, c=0.2),
    "helpful": dict(pp=0.30, mp=1.15, sp=0.5, pm=0.16, mm=0.9, sm=0.4, s0=0.15, b=0.30, c=0.8),
    "weak": dict(pp=0.18, mp=0.6, sp=0.3, pm=0.10, mm=0.5, sm=0.3, s0=0.15, b=0.30, c=0.8),
    # marked-grading shapes (Part A marked: gold Var_true 0.057, sides 178 / 210, chords +-30; teacher 0.033, 279 / 64, +24 / -11)
    "gold_mk": dict(pp=0.30, mp=1.8, sp=0.6, pm=0.42, mm=2.1, sm=0.9, s0=0.15, b=0.08, c=0.2),
    "helpful_mk": dict(pp=0.52, mp=1.8, sp=0.6, pm=0.10, mm=1.1, sm=0.4, s0=0.15, b=0.10, c=0.8),
}
ROUTES = {"catchup": (1.0, 0.0), "catchup75": (0.75, 0.25), "sym": (0.5, 0.5), "fallback75": (0.25, 0.75),
          "fallback": (0.0, 1.0)}
MAJ = {"majcatch": 1.0, "majcatch75": 0.75}  # majority-side catch-up share; minority side symmetric (Part-A-like pattern)
WORLDS = ("catchup", "catchup75", "sym", "fallback75", "fallback", "AR", "null",
          "L:catchup75", "L:sym", "L:fallback75", "L:AR", "AR17", "majcatch", "majcatch75")
TRUE_WORLDS = ("catchup", "catchup75", "L:catchup75")
SIZES = {
    # 17A: 1,319 items, k = 4, untrained 4 x 4, RL alone 3 seeds; RFT 2 seeds (s43 excluded, registered rule)
    "17A": dict(N=1319, k=4, nU=4, theta=(-0.15, 2.3), set_name="gsm8k", rl_seeds=(42, 43, 44), step=375,
                arms=(("rft", "weak", (42, 44), "grpo"), ("hs", "helpful", (42, 43, 44), "grpo"),
                      ("tunc", "weak", (42, 43, 44), "grpo"), ("goldlin", "gold", (42, 43, 44), "grpolin"))),
    # 17B: 1,000 items, k = 8 per checkpoint, untrained k = 24 (3 x 8: same halves 12 / 12), SFT arms 2 seeds, RL alone 3
    "17B": dict(N=1000, k=8, nU=3, theta=(-1.2, 2.3), set_name="math35d", rl_seeds=(42, 43, 44), step=300,
                arms=(("gold", "gold", (42, 43), "grpo"), ("teacher", "helpful", (42, 43), "grpo"),
                      ("rft", "helpful", (42, 43), "grpo"))),
    # marked-grading shapes: lower base success (Part A marked: 27% never solved, 6% always), RL common shift m = 1.27
    # (Part A marked tilt s_U); 17A RFT / T_unc are not weak under marked grading (published marked gains ~ 11 points)
    "17A-mk": dict(N=1319, k=4, nU=4, theta=(-0.9, 2.1), m=1.27, set_name="gsm8k", rl_seeds=(42, 43, 44), step=375,
                   arms=(("rft", "helpful_mk", (42, 44), "grpo"), ("hs", "helpful_mk", (42, 43, 44), "grpo"),
                         ("tunc", "helpful_mk", (42, 43, 44), "grpo"), ("goldlin", "gold_mk", (42, 43, 44), "grpolin"))),
    "17B-mk": dict(N=1000, k=8, nU=3, theta=(-1.95, 2.1), m=1.27, set_name="math35d", rl_seeds=(42, 43, 44), step=300,
                   arms=(("gold", "gold_mk", (42, 43), "grpo"), ("teacher", "helpful_mk", (42, 43), "grpo"),
                         ("rft", "helpful_mk", (42, 43), "grpo"))),
}
MIX_OF = {"17A": {"rft": "weak", "hs": "helpful", "tunc": "weak", "goldlin": "gold"},
          "17B": {"gold": "gold", "teacher": "helpful", "rft": "helpful"},
          "17A-mk": {"rft": "helpful_mk", "hs": "helpful_mk", "tunc": "helpful_mk", "goldlin": "gold_mk"},
          "17B-mk": {"gold": "gold_mk", "teacher": "helpful_mk", "rft": "helpful_mk"}}
SEED_OFF = {"17A": 0, "17B": 50000, "17A-mk": 100000, "17B-mk": 150000}


def draw_delta(rng, N, M):
    u = rng.random(N)
    mp = np.abs(rng.normal(M["mp"], M["sp"], N))
    mm = np.abs(rng.normal(M["mm"], M["sm"], N))
    base = rng.normal(0, M["s0"], N)
    return np.where(u < M["pp"], mp, np.where(u < M["pp"] + M["pm"], -mm, base))


def world18(rng, size, world, m=None, sd_nu=0.5, seed_sd=0.15):
    Z = SIZES[size]
    m = Z.get("m", 0.6) if m is None else m
    assert world != "AR17" or size in ("17A", "17B"), "AR17 uses the registered lenient-shaped 17 specs only"
    N, k, sn = Z["N"], Z["k"], Z["set_name"]
    if world == "AR17":
        if size == "17A":
            spec = S17.arms17a_spec()
            d, _ = S17.ar_world(rng, N, spec, (375,), {"goldlin": "lin", "grpolin": "lin"}, extra_rl={"grpolin": "lin"})
            q = S17.quads17a(d.cells, 375)
            return d, {a: q[a] for a in ("rft", "hs", "tunc", "goldlin")}
        arms = {"gold": dict(c=0.2, sd=1.2, mu_r=-0.5), "teacher": dict(c=0.8, sd=1.0, mu_r=0.2, mu_u=0.3, sd_u=0.3),
                "rft": dict(c=1.0, sd=0.7, mu_r=0.2)}
        d, _ = S17.ar_world(rng, N, arms, (300,), {}, kappa=0.0066, k=8, nU=3, seeds=(42, 43), rl_seeds=S17.SEEDS,
                            set_name="math35d", theta=(-1.2, 2.3))
        return d, S17.quads17b(d.cells, 300)
    lU = rng.normal(*Z["theta"], N) + rng.normal(0, 0.8, N)
    nu = rng.normal(0, sd_nu, N)
    cells = {"U": S17._units(rng, "U", None, [L.expit(lU)] * Z["nU"], k, sn, range(Z["nU"]))}
    cells["U"].lineage = None
    gl = {lin: {s: rng.normal(0, seed_sd, N) for s in Z["rl_seeds"]} for lin in sorted({a[3] for a in Z["arms"]})}
    quads = {}
    fam, kind = ("l", world[2:]) if world.startswith("L:") else ("p", world)
    for arm, mix, seeds, lin in Z["arms"]:
        M = MIXES[mix]
        delta = draw_delta(rng, N, M)
        h = {s: rng.normal(0, seed_sd, N) for s in seeds}
        gp = {s: rng.normal(0, seed_sd, N) for s in seeds}
        w = 0.0 if kind == "null" else 1.0 - M["b"]
        if kind in MAJ:  # catch-up on the arm's majority side only (gold: hurt; helpful: helped), symmetric on its minority side
            k_ = MAJ[kind]
            ap, am = (0.5, 1.0 - k_) if mix.startswith("gold") else (k_, 0.5)
        else:
            ap, am = (M["c"], M["c"]) if kind == "AR" else ROUTES[kind] if kind in ROUTES else (0.5, 0.5)
        alpha = np.where(delta > 0, ap, am)
        pre = [L.expit(lU + delta + h[s]) for s in seeds]
        if fam == "l":
            psi = alpha * delta
            post = [L.expit(lU + delta + h[s] + m + nu + w * (psi - delta - h[s]) + gp[s]) for s in seeds]
            rl = [L.expit(lU + m + nu + w * psi + gl[lin][s]) for s in Z["rl_seeds"]]
        else:
            G = L.expit(lU + delta + m + nu) - L.expit(lU + m + nu)
            post = [np.clip(L.expit(lU + delta + h[s] + m + nu + gp[s]) - w * (1 - alpha) * G, 0, 1) for s in seeds]
            rl = [np.clip(L.expit(lU + m + nu + gl[lin][s]) + w * alpha * G, 0, 1) for s in Z["rl_seeds"]]
        cells[f"{arm}|pre"] = S17._units(rng, f"{arm}|pre", arm, pre, k, sn, seeds)
        cells[f"{arm}|post"] = S17._units(rng, f"{arm}|post", arm, post, k, sn, seeds)
        cells[f"{lin}:{arm}|post"] = S17._units(rng, f"{lin}:{arm}|post", lin, rl, k, sn, Z["rl_seeds"])
        quads[arm] = (f"{arm}|pre", "U", f"{arm}|post", f"{lin}:{arm}|post")
    return L.Design(cells), quads


def calib_one(job):
    size, world, rep, B, nsim = job
    seed = 180000 + SEED_OFF[size] + rep  # common random numbers across worlds
    rng = np.random.default_rng(seed)
    t0 = time.time()
    d, quads = world18(rng, size, world)
    full, reps = L.bootstrap(d, parts_fn(quads, ("len",), ("p",)), B, seed=seed + 7)
    nrng = np.random.default_rng(seed + 11)
    obs, rp, nuls, sU = {}, {}, {"null": {}, "null0": {}}, {}
    for a, q in quads.items():
        obs[a] = table(full, f"len|{a}|")
        rp[a] = table(reps, f"len|{a}|") if B > 0 else None
        if nsim > 0:
            nuls["null"][a], nuls["null0"][a], s, s0 = tilt_null(d, "len", q, nsim, nrng)
            sU[a] = (s, s0)
    if rep == 0:
        for a in quads:
            check_against_frozen(obs[a])
    rec = summarise(obs, rp if B > 0 else None, nuls if nsim > 0 else None, list(quads))
    rec.update(size=size, world=world, rep=rep, sU=sU, secs=round(time.time() - t0, 1))
    return rec


def aggregate(recs, margins=(0.0, 0.1, 0.2)):
    sizes = sorted({r["size"] for r in recs})
    for size in sizes:
        R = [r for r in recs if r["size"] == size]
        worlds = [w for w in WORLDS if any(r["world"] == w for r in R)]
        arms = list(R[0]["arms"])
        P(f"\n## {size}: {len(R)} datasets; worlds {worlds}; arms {arms} (mix: "
          + ", ".join(f"{a} {MIX_OF[size].get(a, '?')}" for a in arms) + "; AR17 = score_prereg17.ar_world specs)")
        P("\n### descriptives (means over datasets; lenient synthetic; probability scale)")
        for w in worlds:
            Rw = [r for r in R if r["world"] == w]
            P(f"  [{w}] n = {len(Rw)}; mean secs {np.mean([r['secs'] for r in Rw]):.0f}")
            for a in arms:
                A = [r["arms"][a] for r in Rw]
                line = (f"    {a:8s} Var_true {np.nanmean([x['VD'] for x in A]):.4f} b {np.nanmean([x['b'] for x in A]):+.3f} "
                        f"gap {np.nanmean([x['gap_pts'] for x in A]):+5.1f} pts |")
                for s in SIDES:
                    X = [x[s] for x in A]
                    g = lambda k: np.nanmean([x[k] for x in X])
                    e1 = np.array([x["d"] - x["d_null"] for x in X], float)
                    e0 = np.array([x["d"] - x["d_null0"] for x in X], float)
                    line += (f" {s}: n {g('n'):5.1f} chord {g('chord'):+5.1f} 1-b {g('1-b'):+.2f} U {g('U'):+.2f} C {g('C'):+.2f} "
                             f"d {g('d'):+.2f} | exc(all-tilt) {np.nanmean(e1):+.2f}±{np.nanstd(e1):.2f} | exc(ref-tilt) {np.nanmean(e0):+.2f}±{np.nanstd(e0):.2f} ")
                    if "Ld" in X[0]:
                        l0 = np.array([x["Ld"] for x in X], float)
                        l1 = np.array([x["Ld"] - x["Ld_null"] for x in X], float)
                        line += f"| LM d {np.nanmean(l0):+.2f}±{np.nanstd(l0):.2f} null {g('Ld_null'):+.2f} exc {np.nanmean(l1):+.2f}±{np.nanstd(l1):.2f} "
                    line += "| testable " + "/".join(f"{np.mean([x['test'][gk] for x in X]):.2f}" for gk in GATES) + " |"
                P(line)
            P("    tilt s (all items, reference group) per arm: " + "; ".join(
                f"{a} {np.mean([r['sU'][a][0] for r in Rw]):+.2f}, {np.mean([r['sU'][a][1] for r in Rw]):+.2f}" for a in arms if a in Rw[0]["sU"]))
            for gk in (REG18["GATE"], "ungated"):
                for name, _ in Rw[0]["pools"][gk].items():
                    X = [r["pools"][gk][name] for r in Rw]
                    e1 = np.array([x["d"] - x["d_null"] for x in X], float)
                    e0 = np.array([x["d"] - x["d_null0"] for x in X], float)
                    P(f"    pool {gk}:{name:9s} raw d {np.nanmean([x['d'] for x in X]):+.2f} | exc(all-tilt) {np.nanmean(e1):+.2f} SD {np.nanstd(e1):.2f} "
                      f"| exc(ref-tilt) {np.nanmean(e0):+.2f} SD {np.nanstd(e0):.2f} | CI width {np.nanmean([x['d_ci'][1] - x['d_ci'][0] for x in X]):.2f} "
                      f"(arm-sides in pool {np.mean([len(x['items']) for x in X]):.1f})")
        variants = [(gk, ex, 0.0, "d") for gk in GATES for ex in MODES] + [(REG18["GATE"], ex, m, "d") for ex in NULLS for m in margins if m > 0]
        if "Ld" in R[0]["arms"][arms[0]]["+"]:
            variants += [(REG18["GATE"], ex, 0.0, "Ld") for ex in MODES]
        for gk, excess, mg, est in variants:
            tag = ESTS[est] + ", " + MODES[excess] + f", gate {gk} {GATES[gk]}, point margin {mg:.1f}"
            P(f"\n### {size} P(held | world) — {tag}. Untestable counts as not held; (u.. = share untestable)")
            rr = {w: [rules(r, excess, mg, gk, est) for r in R if r["world"] == w] for w in worlds}
            keys = [k for k in rr[worlds[0]][0] if k != "n_testable"]
            P("    rule".ljust(24) + "".join(f"{w:>13s}" for w in worlds))
            for k in keys:
                cols = ""
                for w in worlds:
                    v = [x.get(k) for x in rr[w]]
                    held = np.mean([x is True for x in v])
                    un = np.mean([x is None for x in v])
                    cols += f"{held:7.2f}" + (f"(u{un:.2f})"[:6] if un > 0.005 else "      ")
                P(f"    {k:20s}" + cols)
            P("    mean n_testable".ljust(24) + "".join(f"{np.mean([x['n_testable'] for x in rr[w]]):13.2f}" for w in worlds))


def point_rule(rec, mode, kind, t1, t2, g=None, est="d"):
    """Point-threshold rules on pooled excess contrasts (thresholds calibrated by simulation, as prereg 15's SPLIT_D):
    'MM': majority-side pool > t1 AND minority-side pool > t2; 'PM': plus pool > t1 AND minus pool > t2;
    'MMci': majority pool CI lower bound > 0 AND minority pool point > t2. None if a pool has no testable arm-side.
    est 'd': frozen chord; 'Ld': location-matched chord."""
    Pp = rec["pools"][g or REG18["GATE"]]
    if kind == "DISC":  # the discriminating test alone: a priori disc pool > t2 (t1 unused)
        return minority_rule(rec, mode, est, g or REG18["GATE"], t2, False, "disc")
    a, b = ("major", "minor") if kind.startswith("MM") else ("agree", "disc") if kind.startswith("AD") else ("plus", "minus")
    if not Pp[a]["items"] or not Pp[b]["items"] or not (np.isfinite(Pp[a][est]) and np.isfinite(Pp[b][est])):
        return None
    if kind in ("MM2", "AD2") and len(Pp[a]["items"]) < 2:  # 'MM2': the majority pool must hold >= 2 testable arm-sides
        return None
    xa, la = _dl(Pp[a], mode, est)
    xb, _ = _dl(Pp[b], mode, est)
    if kind == "MMci":
        return bool(la > 0 and xb > t2)
    return bool(xa > t1 and xb > t2)


GRID = [("MM", t1, t2) for t1 in (0.0, 0.05, 0.10, 0.15, 0.20) for t2 in (-0.10, 0.0, 0.10)] + \
       [("MMci", 0.0, t2) for t2 in (-0.10, 0.0, 0.10)] + [("PM", t, t) for t in (0.0, 0.05, 0.10, 0.15)] + \
       [("MM2", t1, t2) for t1 in (0.10, 0.15, 0.20) for t2 in (-0.15, -0.10, -0.05, 0.0, 0.05)] + \
       [("AD2", t1, t2) for t1 in (0.10, 0.15, 0.20) for t2 in (-0.15, -0.10, -0.05, 0.0, 0.05)]


def grid(recs):
    for size in sorted({r["size"] for r in recs}):
        R = [r for r in recs if r["size"] == size]
        worlds = [w for w in WORLDS if any(r["world"] == w for r in R)]
        ests = [e for e in ESTS if e in R[0]["arms"][next(iter(R[0]["arms"]))]["+"]]
        for est in ests:
            for gk in ("n40c3", "n40"):
                for mode in (False,) + NULLS:
                    P(f"\n### {size} point-threshold rules — {ESTS[est]}, {MODES[mode]}, gate {gk}: P(held | world) over "
                      f"{len(R) // len(worlds)} datasets (MM: majority pool > t1 and minority pool > t2; PM: plus pool > t1 and "
                      f"minus pool > t2; MMci: majority CI > 0)")
                    P("    rule (t1, t2)".ljust(26) + "".join(f"{w:>13s}" for w in worlds))
                    for kind, t1, t2 in GRID:
                        cols = ""
                        for w in worlds:
                            v = [point_rule(r, mode, kind, t1, t2, gk, est) for r in R if r["world"] == w]
                            cols += f"{np.mean([x is True for x in v]):13.2f}"
                        P(f"    {kind:5s} ({t1:+.2f}, {t2:+.2f})".ljust(26) + cols)
            for mode in (False,) + NULLS:
                P(f"\n### {size} per arm-side ({ESTS[est]}, {MODES[mode]}): share testable (gate n40c3) / P(contrast > 0 | testable)")
                arms = list(R[0]["arms"])
                P("    arm-side".ljust(26) + "".join(f"{w:>13s}" for w in worlds))
                for a in arms:
                    for s in SIDES:
                        cols = ""
                        for w in worlds:
                            X = [r["arms"][a][s] for r in R if r["world"] == w]
                            t = [x for x in X if x["test"]["n40c3"]]
                            pos = np.mean([_dl(x, mode, est)[0] > 0 for x in t]) if t else np.nan
                            cols += f"   {len(t) / len(X):4.2f}/{pos:4.2f}"
                        P(f"    {a}|{s}".ljust(26) + cols)
            P(f"\n### {size} pooled contrasts ({ESTS[est]}, gate n40c3): mean (SD over datasets) of raw and excess values")
            P("    pool / version".ljust(26) + "".join(f"{w:>13s}" for w in worlds))
            for name in ("plus", "minus", "minor", "major"):
                for mode in (False,) + NULLS:
                    cols = ""
                    for w in worlds:
                        v = np.array([_dl(r["pools"]["n40c3"][name], mode, est)[0] for r in R if r["world"] == w
                                      and r["pools"]["n40c3"][name]["items"]], float)
                        cols += f"  {np.nanmean(v):+5.2f}({np.nanstd(v):4.2f})" if v.size else "          nan"
                    P(f"    {name} {('raw' if not mode else mode)}".ljust(26) + cols)


MIN_T = (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10)


def minority_rule(rec, mode, est, g, t=None, ci=False, pool="disc"):
    """The DISCRIMINATING test on its own: the pooled contrast over the discriminating sides > t (point), or with its
    95% CI lower bound > 0 (ci=True). pool 'disc': a priori sides (disc_side: helped side of gold-type arms, hurt side of
    the others); pool 'minor': each testable arm's smaller observed side. Per-item catch-up predicts > 0 there; 'the
    route follows the arm' (AR) predicts a clearly negative value. None = no testable side in the pool."""
    x = rec["pools"][g].get(pool)
    if x is None or not x["items"] or not np.isfinite(x[est]):
        return None
    d, lo = _dl(x, mode, est)
    return bool(lo > 0) if ci else bool(d > t)


def minority_table(recs):
    for size in sorted({r["size"] for r in recs}):
        R = [r for r in recs if r["size"] == size]
        worlds = [w for w in WORLDS if any(r["world"] == w for r in R)]
        ests = [e for e in ESTS if e in R[0]["pools"]["n40c3"]["minor"]]
        pools = [p for p in ("disc", "minor") if p in R[0]["pools"]["n40c3"]]
        for pool in pools:
            for est in ests:
                for gk in ("n40", "n40c3"):
                    lab0 = "A PRIORI DISCRIMINATING-SIDE TEST (gold helped + helpful hurt)" if pool == "disc" else \
                        "MINORITY-SIDE TEST (smaller observed side per arm)"
                    P(f"\n### {size} {lab0} — {ESTS[est]}, gate {gk}: P(held | world) over {len(R) // len(worlds)} datasets "
                      f"(pooled contrast > t, or 95% CI lower bound > 0); untestable = no testable side in the pool")
                    P("    version, threshold".ljust(30) + "".join(f"{w:>13s}" for w in worlds))
                    un = [np.mean([not r["pools"][gk][pool]["items"] for r in R if r["world"] == w]) for w in worlds]
                    P("    (untestable share)".ljust(30) + "".join(f"{u:13.2f}" for u in un))
                    for mode in (False,) + NULLS:
                        for t in MIN_T + ("CI",):
                            cols = ""
                            for w in worlds:
                                v = [minority_rule(r, mode, est, gk, None if t == "CI" else t, t == "CI", pool)
                                     for r in R if r["world"] == w]
                                cols += f"{np.mean([x is True for x in v]):13.2f}"
                            lab = "CI lower > 0" if t == "CI" else f"> {t:+.2f}"
                            P(f"    {('raw' if not mode else mode):5s} {lab}".ljust(30) + cols)


def dry_synthetic(args):
    from multiprocessing import Pool
    sizes = args.sizes.split(",")
    worlds = args.worlds.split(",")
    jobs = [(sz, w, r, args.B, args.nsim) for sz in sizes for r in range(args.reps) for w in worlds]
    P(f"# DRY RUN 18 synthetic: {len(jobs)} datasets (sizes {sizes}, worlds {worlds}, {args.reps} per world and size, "
      f"B {args.B}, tilt-null sims {args.nsim})\n# MIXES {MIXES}\n# ROUTES {ROUTES}")
    out = Path(args.calib_out)
    recs = []
    with Pool(args.procs) as pool, open(out, "w") as f:
        for i, rec in enumerate(pool.imap_unordered(calib_one, jobs, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            recs.append(rec)
            if (i + 1) % 50 == 0:
                P(f"  {i + 1}/{len(jobs)} datasets done")
    P(f"# per-dataset records: {out}")
    aggregate(recs)
    grid(recs)
    minority_table(recs)


# ================================================================ real data: reporting
def report(label, obs, rep, nuls, arms, sets, scales=SCALES):
    """Print every statistic for single arms and pooled sets, both scales (location-matched chord: probability scale);
    return the probability-scale records."""
    recs = {}
    for sc in scales:
        P(f"\n## {label} — scale {'probability (primary)' if sc == 'p' else 'smoothed logit (reported)'}")
        fams = [f for f in FAMS if f"{sc}|{f[0]}|p_dS" in obs[arms[0]]]
        for unit, members in [(a, (a,)) for a in arms] + list(sets.items()):
            o_ = [obs[a] for a in members]
            r_ = [rep[a] for a in members] if rep is not None else None
            n_ = {nk: [nuls[nk][a] for a in members] for nk in NULLS} if nuls is not None else None
            if sc == "p":
                e, er = eiv_stats(o_), eiv_stats(r_) if r_ else None
                P(f"  {unit} ({'+'.join(members)}): Var_true(D_pre) {fmt(e['VDpre'][0], L.ci(er['VDpre']) if er else None, 4)}; "
                  f"b {fmt(e['b'][0], L.ci(er['b']) if er else None)}; beta_S {fmt(e['beta_S'][0])}; beta_U {fmt(e['beta_U'][0])}; "
                  f"gap pre {e['gap_pts'][0]:+.2f} pts")
            else:
                P(f"  {unit} ({'+'.join(members)}):")
            for side in SIDES:
                for fam, pre in fams:
                    so = side_stats(o_, side, sc, fam)
                    sr = side_stats(r_, side, sc, fam) if r_ else None
                    sn = {nk: side_stats(v, side, sc, fam) for nk, v in n_.items()} if n_ else None
                    c = lambda k: L.ci(sr[k]) if sr else None
                    nm = lambda nk, k: med(sn[nk][k]) if sn else NAN
                    n, omb = float(so["n"][0]), float(so["1-b"][0])
                    lab = "helped" if side == "+" else "hurt"
                    if fam == "W":
                        tag = ("testable" if gate(n, omb, float(so["chord"][0])) else "UNTESTABLE") + f" under gate {REG18['GATE']} {GATES[REG18['GATE']]}"
                        P(f"    side {side} ({lab}): items per arm {n:.1f}; chord D_pre {so['chord'][0]:+.2f} pts; 1-b{side} {fmt(so['1-b'][0], c('1-b'))} [{tag}]")
                    else:
                        P(f"    side {side} ({lab}), LOCATION-MATCHED chord: items per arm {n:.1f}; chord {so['chord'][0]:+.2f} pts; "
                          f"1-b{side} {fmt(so['1-b'][0], c('1-b'))}")
                    P(f"      1-b{side}: undoing part U{side} {fmt(so['U'][0], c('U'))}; convergence part C{side} {fmt(so['C'][0], c('C'))}"
                      f" | all-items tilt null U {nm('null', 'U'):+.3f} C {nm('null', 'C'):+.3f} | reference-group tilt null U "
                      f"{nm('null0', 'U'):+.3f} C {nm('null0', 'C'):+.3f}")
                    dd = so["d"][0]
                    ci_ = c("d")
                    line = f"      {'LM ' if fam == 'LM' else ''}d{side} = {'C+ - U+' if side == '+' else 'U- - C-'} {fmt(dd, ci_)}"
                    for nk, nlab in (("null", "all-items tilt"), ("null0", "reference-group tilt")):
                        m0 = nm(nk, "d")
                        line += f" | {nlab}: median {m0:+.3f}, excess {fmt(dd - m0, (ci_[0] - m0, ci_[1] - m0) if ci_ else None)}"
                    P(line)
        if sc == "p":
            for unit, members in [("arms", tuple(arms))] + list(sets.items()):
                recs[unit] = summarise(obs, rep, nuls, list(members), sc)
    return recs


def verdict(v):
    return "HELD" if v is True else "not held" if v is False else "untestable"


def print_rules(label, rec):
    for est in [e for e in ESTS if e in rec["pools"][REG18["GATE"]]["major"]]:
        for excess in MODES:
            r = rules(rec, excess, REG18["MARGIN"], None, est)
            P(f"  [{label}] rules, {ESTS[est]}, {MODES[excess]}: " + "; ".join(f"{k} {verdict(v)}" for k, v in r.items() if k != "n_testable")
              + f"; testable arm-sides {r['n_testable']}")
        Pp = rec["pools"][REG18["GATE"]]
        for name in ("agree", "disc", "major", "minor", "plus", "minus"):
            x = Pp[name]
            P(f"  [{label}] {ESTS[est]} pool {name:5s} (gate {REG18['GATE']}; arm-sides {x['items']}): raw {fmt(x[est], tuple(x[est + '_ci']))}; "
              + "; ".join(f"excess over {nk} {fmt(x[est] - x[f'{est}_{nk}'], (x[est + '_ci'][0] - x[f'{est}_{nk}'], x[est + '_ci'][1] - x[f'{est}_{nk}']))}" for nk in NULLS))
    for tag, (kind, est, mode, t1, t2) in [("DISCRIMINATING", REG18["DISCRIMINATING"]), ("PRIMARY", REG18["PRIMARY"])] + \
            [("secondary", s) for s in REG18["SECONDARY"]]:
        v = point_rule(rec, mode, kind, t1, t2, None, est)
        P(f"  [{label}] {tag} {kind}(t1 {t1:+.2f}, t2 {t2:+.2f}), {ESTS[est]}, {MODES[mode]}, gate {REG18['GATE']}: {verdict(v)}")


def run_real(label, d, quads, graders, steps_arms, sets, args, bseed, nseed):
    """quads: {f'{step}|{arm}': quad}. Bootstrap (frozen), tilt nulls (frozen pieces, full fit), report and rules."""
    t0 = time.time()
    full, reps = L.bootstrap(d, parts_fn(quads, graders), args.B, seed=bseed)
    P(f"# bootstrap B = {args.B} done ({time.time() - t0:.0f}s)")
    rng = np.random.default_rng(nseed)
    out = {}
    for o in graders:
        nul, nul0 = {}, {}
        for lab, q in quads.items():
            nul[lab], nul0[lab], sU, s0 = tilt_null(d, o, q, args.nsim, rng, full_fit=True)
            P(f"   tilt nulls [{o}] {lab}: s_U all items {sU:+.3f}, reference group {s0:+.3f} ({time.time() - t0:.0f}s)")
        for step, arms in steps_arms.items():
            labs = [f"{step}|{a}" for a in arms if f"{step}|{a}" in quads]
            obs = {lab.split("|", 1)[1]: table(full, f"{o}|{lab}|") for lab in labs}
            rp = {lab.split("|", 1)[1]: table(reps, f"{o}|{lab}|") for lab in labs}
            nl = {"null": {lab.split("|", 1)[1]: nul[lab] for lab in labs}, "null0": {lab.split("|", 1)[1]: nul0[lab] for lab in labs}}
            for a in obs:
                check_against_frozen(obs[a])
            ss = {k: tuple(a for a in v if a in obs) for k, v in sets.items()}
            ss = {k: v for k, v in ss.items() if v}
            recs = report(f"{label} [{o}] step {step}", obs, rp, nl, list(obs), ss)
            P(f"\n## {label} [{o}] step {step}: candidate rules (probability scale, gate n >= {REG18['MIN_SIDE_N']} and 1-b >= {REG18['GATE_1MB']})")
            for unit, rec in recs.items():
                if unit != "arms":
                    print_rules(f"{o} {step} {unit}", rec)
            out[(o, step)] = {"recs": recs, "obs": obs, "rep": rp, "nul": nl}
    return out


# ================================================================ reference: prereg-15 Part A (already scored)
PUBLISHED_A = {  # runs/theory/scratch/score_prereg15_partA_out.txt, [len] step 375, probability scale: (point, lo, hi)
    "gold_pool": {"U+": (0.283, 0.011, 0.440), "C+": (0.534, 0.350, 0.781), "U-": (0.848, 0.781, 0.976), "C-": (0.095, -0.033, 0.186)},
    "teacher_pool": {"U+": (-0.194, -0.696, 0.011), "C+": (0.873, 0.648, 1.213), "U-": (0.626, 0.395, 1.025), "C-": (0.107, -0.466, 0.499)},
}


def reference_partA(args):
    paths = S.Paths(str(REPO), "runs/gsm8k/Llama-3.2-1B-Instruct-px-{proto}{src}")
    P("\n# Reference run: prereg-15 Part A (S' gold / teacher, step 375), frozen loaders S.load_A / S.part_A_exclusions / S.cells_A")
    ev, unt, e2u = S.load_A(paths)
    bad, _ = S.part_A_exclusions(paths)
    P("excluded runs:", sorted(bad) or "none")
    d = L.Design(S.cells_A(ev, unt, e2u, bad))  # the identical design (all arms, all steps) -> identical bootstrap draws
    arms = ("gold_250", "gold_500", "teacher_250", "teacher_500")
    quads = {f"375|{a}": S.quads([a], 375)[0] for a in arms}
    sets = {"gold_pool": S.POOLS["gold_pool"], "teacher_pool": S.POOLS["teacher_pool"], "all_four": arms}
    res = run_real("Part A", d, quads, L.GRADERS, {375: arms}, sets, args, bseed=1501, nseed=1802)
    P("\n## Reference match (lenient, probability scale, step 375): published (frozen scorer; score_prereg15_partA_out.txt) "
      "vs this run (same design and bootstrap seed 1501, so CIs should agree too)")
    R = res[("len", 375)]
    worst = 0.0
    for g, pub in PUBLISHED_A.items():
        mem = sets[g]
        for k, (v, lo, hi) in pub.items():
            part, side = k[0], k[1]
            x = float(side_stats([R["obs"][a] for a in mem], side)[part][0])
            clo, chi = L.ci(side_stats([R["rep"][a] for a in mem], side)[part])
            worst = max(worst, abs(x - v), abs(clo - lo), abs(chi - hi))
            P(f"  {g:12s} {k}: this run {fmt(x, (clo, chi))} | published {fmt(v, (lo, hi))} | |diff| point {abs(x - v):.4f}, "
              f"CI {abs(clo - lo):.4f} / {abs(chi - hi):.4f}")
    P(f"  largest absolute difference {worst:.4f} (published values are rounded to 0.001) -> "
      + ("REPRODUCED" if worst <= 0.0006 else "DIFFERS"))
    return res


# ================================================================ real data: 17A / 17B (unblind)
class _Captured(Exception):
    pass


def capture_designs(fn, paths, args):
    """Run the frozen part_17a / part_17b up to their bootstrap call and return the Designs they build (cells and
    registered exclusions exactly as registered). Their bootstrap, nulls and decisions are stubbed out for the call."""
    designs = []
    saved = (L.bootstrap, S.run_nulls, S17.decide17a, S17.decide17b)

    def boot(design, fn_, B, seed, seeds=True):
        designs.append(design)
        return {}, {}
    try:
        L.bootstrap = boot
        S.run_nulls = lambda *a, **k: ({}, {})
        S17.decide17a = lambda *a, **k: {}
        S17.decide17b = lambda *a, **k: {}
        fn(paths, SimpleNamespace(B=0, nsim=0, proto=args.proto))
    finally:
        L.bootstrap, S.run_nulls, S17.decide17a, S17.decide17b = saved
    return designs


def relabel(design, prefix):
    cells = {}
    for c in design.cells.values():
        n = L.Cell(prefix + c.name, c.set, None if c.lineage is None else prefix + c.lineage, c.seeds, c.C, c.Nh)
        n.ids = getattr(c, "ids", None)
        cells[n.name] = n
    return cells


def unblind_17a(paths, args):
    designs = capture_designs(S17.part_17a, paths, args)
    assert len(designs) == 1
    d = designs[0]
    steps = [int(s) for s in args.steps17a.split(",")]
    quads = {}
    for t in steps:
        for a, q in S17.quads17a(d.cells, t).items():
            quads[f"{t}|{a}"] = q
    P("quads:", {k: v for k, v in quads.items()})
    arms = ("rft", "hs", "tunc", "goldlin", "gold_250", "teacher_250")
    sets = {"17A_new": ("rft", "hs", "tunc", "goldlin"), "17A_helpful": ("rft", "hs", "tunc"), "S'_ref": ("gold_250", "teacher_250")}
    return run_real("17A", d, quads, L.GRADERS, {t: arms for t in steps}, sets, args, bseed=1801, nseed=1802)


def unblind_17b(paths, args):
    designs = capture_designs(S17.part_17b, paths, args)
    fams = [f for f in S17.FAMILIES]
    if len(designs) != len(fams):
        P(f"WARNING: {len(designs)} family design(s) built (a family without its untrained evaluation is skipped by the frozen loader)")
    # part_17b iterates FAMILIES in order and skips a family only when its untrained file is missing
    present = [f for f in fams if (paths.root / S17.P17B / f"{f}_{args.proto}_untrained.jsonl").exists()]
    assert len(present) == len(designs)
    cells, quads = {}, {}
    ids0 = None
    for fam, d in zip(present, designs):
        ids = d.cells["U"].ids
        ids0 = ids if ids0 is None else ids0
        assert ids == ids0, "17B families scored on different item lists"
        cells.update(relabel(d, f"{fam}:"))
        for t in S17.STEPS17B:
            for a, q in S17.quads17b(d.cells, t).items():
                quads[f"{t}|{fam}:{a}"] = tuple(f"{fam}:{x}" for x in q)
    D = L.Design(cells)  # one design: items resampled jointly (the same 1,000 test items), lineages per family
    P("quads:", quads)
    arms = tuple(f"{f}:{a}" for f in present for a in S17.ARMS17B)
    sets = {f"{f}_all": tuple(f"{f}:{a}" for a in S17.ARMS17B) for f in present}
    sets.update({f"{f}_helpful": (f"{f}:teacher", f"{f}:rft") for f in present})
    if len(present) == 2:
        sets["both_all"] = arms
    return run_real("17B", D, quads, L.GRADERS, {t: arms for t in (300, 150)}, sets, args, bseed=1811, nseed=1812)


# ================================================================ schema-only (no outcome field is read)
def schema_read(path):
    """Tolerant schema pass: per adapter, rows / items / samples and the key set; a truncated last line (file still
    being written) is counted, not parsed. Reads only 'adapter', 'example_id', 'sample' values."""
    info = defaultdict(lambda: {"rows": 0, "items": set(), "samples": Counter()})
    keys, bad = set(), 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            x = info[r.get("adapter")]
            x["rows"] += 1
            x["items"].add(int(r["example_id"]))
            x["samples"][int(r["sample"])] += 1
            keys |= set(r)
    return info, keys, bad


_RX17B = r"/math35d/{fam}-p17b-{proto}-(grpo|gold|teacher|rft)/seed_(\d+)/(listed_sft|full_grpo)/(sft|grpo)(?:/checkpoint-(\d+))?/?$"


def schema18(paths, part, proto):
    import re
    P(f"# schema-only, part {part}: file presence, adapters, rows, items, samples per item; NO outcome field is read")
    if part in ("17A", "all"):
        P("\n## 17A probe files (runs/probes/prereg17a)")
        have = defaultdict(set)
        for f in sorted((paths.root / S17.P17A).glob("*.jsonl")):
            info, keys, bad = schema_read(f)
            P(f"  {f.name}: {len(info)} adapters; rows/adapter {sorted({x['rows'] for x in info.values()})}; "
              f"items/adapter {sorted({len(x['items']) for x in info.values()})}; samples/item {sorted({max(x['samples']) + 1 for x in info.values()})}; "
              f"unparsable lines {bad}; keys {sorted(keys)}")
            for a in info:
                arm, seed, stage = S17.parse17a(a)
                have[(arm, stage)].add(seed)
        P("  (arm, stage) -> seeds present in the 17A probe files:")
        for k in sorted(have, key=str):
            P(f"    {k}: {sorted(have[k])}")
        P("  prereg-18 17A quads at step 375 (cells need >= 2 seeds; the registered stop-rate exclusion — RFT seed 43 per "
          "prereg 17 Outcome — is applied by the frozen loader at unblinding):")
        for arm, (pre, post, comp) in {"rft": (("rft", "pre"), ("rft", 375), ("grpo", 375)),
                                       "hs": (("hs", "pre"), ("hs", 375), ("grpo", 375)),
                                       "tunc": (("tunc", "pre"), ("tunc", 375), ("grpo", 375)),
                                       "goldlin": (("gold_250", "pre"), ("goldlin", 375), ("grpolin", 375))}.items():
            src = {k: sorted(have.get(k, set())) for k in (pre, post, comp)}
            note = " (gold_250|pre and grpo|375 come from prereg-14 Part A files, not the 17A directory)" if arm == "goldlin" or comp[0] == "grpo" else ""
            P(f"    {arm}: SFT start {pre} seeds {src[pre] or 'Part A'}; post {post} seeds {src[post]}; comparator {comp} seeds {src[comp] or 'Part A'}{note}")
    if part in ("17B", "all"):
        P(f"\n## 17B (protocol {proto})")
        for fam in S17.FAMILIES:
            rx = re.compile(_RX17B.format(fam=fam, proto=proto))
            have = defaultdict(set)
            for tag in S17.STAGE_FILES17B:
                f = paths.root / S17.P17B / f"{fam}_{proto}_{tag}.jsonl"
                if not f.exists():
                    P(f"  {f.name}: MISSING")
                    continue
                info, keys, bad = schema_read(f)
                comp = []
                for a, x in info.items():
                    k = max(x["samples"]) + 1
                    complete = x["rows"] == len(x["items"]) * k and len(x["items"]) == 1000
                    if a is None:
                        key = ("U", 0, "pre")
                    else:
                        m = rx.search(a)
                        key = ("UNPARSED", a, None) if m is None else (m.group(1), int(m.group(2)), "pre" if m.group(4) == "sft" else int(m.group(5)))
                    comp.append((key, x["rows"], len(x["items"]), k, complete))
                    if complete:
                        have[(key[0], key[2])].add(key[1])
                P(f"  {f.name}: {len(info)} adapters; unparsable lines {bad}; keys {sorted(keys)}")
                for key, rows, ni, k, ok in sorted(comp, key=str):
                    P(f"    {key}: rows {rows}, items {ni}, samples/item {k} -> {'complete' if ok else 'INCOMPLETE (being written?)'}")
            P(f"  {fam}: complete cells (arm, stage) -> seeds: " + "; ".join(f"{k}: {sorted(v)}" for k, v in sorted(have.items(), key=str)))
            for t in S17.STEPS17B:
                for arm in S17.ARMS17B:
                    need = {(arm, "pre"): S17.SEEDS17B[arm], (arm, t): S17.SEEDS17B[arm], ("grpo", t): S17.SEEDS17B["grpo"]}
                    st = {k: sorted(have.get(k, set())) for k in need}
                    okq = ("U", "pre") in have and all(len(v) >= 2 for v in st.values())
                    P(f"    quad {fam} {arm} @ {t}: " + ", ".join(f"{k} {v}" for k, v in st.items()) + f" -> {'formable' if okq else 'NOT formable yet'}")
            for arm in ("grpo",) + S17.ARMS17B:
                for s in S17.SEEDS17B[arm]:
                    base = S17.run_dir17b(paths.root, fam, proto, arm) / f"seed_{s}"
                    meth = "full_grpo" if arm == "grpo" else "listed_sft"
                    fs = ([] if arm == "grpo" else [f"{meth}/sft/adapter_model.safetensors"]) + \
                         [f"{meth}/grpo/checkpoint-{t}/adapter_model.safetensors" for t in S17.STEPS17B] + [f"{meth}/grpo/rollouts.jsonl"]
                    P(f"    training {fam} {arm} seed {s}: " + ", ".join(
                        f"{Path(x).parent.name}/{Path(x).name} {'ok' if (base / x).exists() else 'MISSING'}" for x in fs))


# ================================================================ mock end-to-end (unblind code path on synthetic files)
def make_mock(root):
    """A mock repository root for the 17A and 17B unblind paths: real prereg-14 Part A inputs symlinked (already
    scored, allowed), SYNTHETIC 17A new-arm probe files (all 1,319 items, k = 4; pre and step 375 only) with a planted
    catch-up world, a synthetic rollouts file that triggers the registered RFT exclusion, and SYNTHETIC 17B files for
    both families (120 items, numeric gold, k = 8 / 24)."""
    root = Path(root)
    assert not root.exists(), f"{root} exists"
    rng = np.random.default_rng(1818)
    (root / "prereg").mkdir(parents=True)
    runs = root / "runs"
    for rel in ("runs/probes/prereg14_a", "runs/gsm8k/Llama-3.2-1B-Instruct-marked-steps375",
                "runs/gsm8k/Llama-3.2-1B-Instruct-marked-steps375-qwen7b", "runs/gsm8k/Llama-3.2-1B-Instruct-censored-grpo"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).symlink_to(REPO / rel)
    # 17A: ids and gold from the real untrained Part A file (schema fields only)
    ids, gold = [], {}
    with open(REPO / "runs/probes/prereg14_a/untrained.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if int(r["sample"]) == 0:
                ids.append(int(r["example_id"]))
                gold[int(r["example_id"])] = r["gold"]
    ids = sorted(ids)
    N = len(ids)
    lU = rng.normal(-0.15, 2.3, N) + rng.normal(0, 0.8, N)
    G = "/orcd/mock/runs/gsm8k/Llama-3.2-1B-Instruct-p17a-{arm}/seed_{s}/{meth}/{stage}"
    meth = {"rft": "random_sft_250", "hs": "random_sft_250", "tunc": "listed_sft", "goldlin": "random_sft_250", "grpolin": "full_grpo"}
    (runs / "probes/prereg17a").mkdir(parents=True)

    def rows17a(fh, adapter, p, k=4):
        for j, e in enumerate(ids):
            for s in range(k):
                ok = rng.random() < p[j]
                fh.write(json.dumps({"adapter": adapter, "example_id": e, "sample": s, "gold": gold[e], "correct": bool(ok),
                                     "stopped": True, "completion": f"Reasoning.\n#### {gold[e] if ok else '987654'}"}) + "\n")
    with open(runs / "probes/prereg17a/new_sft.jsonl", "w") as fs_, open(runs / "probes/prereg17a/new_s375.jsonl", "w") as fp_:
        for arm in ("rft", "hs", "tunc", "goldlin", "grpolin"):
            delta = draw_delta(rng, N, MIXES["weak" if arm in ("rft", "tunc") else "helpful"]) if arm != "grpolin" else np.zeros(N)
            psi = np.where(delta > 0, delta, 0.0)  # catch-up
            for s in (42, 43, 44):
                if arm in ("rft", "hs", "tunc"):
                    rows17a(fs_, G.format(arm=arm, s=s, meth=meth[arm], stage="sft"), L.expit(lU + delta))
                rows17a(fp_, G.format(arm=arm, s=s, meth=meth[arm], stage="grpo/checkpoint-375"),
                        L.expit(lU + 0.6 + delta + 0.8 * (psi - delta)))
    for arm, s, stop in (("rft", 43, False), ("rft", 42, True)):
        f = runs / f"gsm8k/Llama-3.2-1B-Instruct-p17a-{arm}/seed_{s}/{meth[arm]}/grpo/rollouts.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fh:
            for step in range(375):
                fh.write(json.dumps({"step": step, "reward": 1.0, "stopped": stop or step < 300}) + "\n")
    # 17B: synthetic, numeric gold answers, boxed completions graded by the pipeline's math35d grader
    (runs / "probes/prereg17b").mkdir(parents=True)
    n17 = 120
    for fam in S17.FAMILIES:
        l0 = rng.normal(-1.2, 2.3, n17)
        dl = {a: draw_delta(rng, n17, MIXES["gold" if a == "gold" else "helpful"]) for a in S17.ARMS17B}
        B = f"/orcd/mock/runs/math35d/{fam}-p17b-mf-{{arm}}/seed_{{s}}/{{meth}}/{{stage}}"

        def rows17b(fh, adapter, p, k):
            for e in range(n17):
                for s in range(k):
                    ok = rng.random() < p[e]
                    g = str(10 + e)
                    fh.write(json.dumps({"adapter": adapter, "example_id": e, "id": f"x{e}", "sample": s, "gold": g,
                                         "correct": bool(ok), "stopped": True,
                                         "completion": f"So \\boxed{{{g if ok else '99999'}}}"}) + "\n")
        with open(runs / f"probes/prereg17b/{fam}_mf_untrained.jsonl", "w") as fh:
            rows17b(fh, None, L.expit(l0), 24)
        with open(runs / f"probes/prereg17b/{fam}_mf_sft.jsonl", "w") as fh:
            for a in S17.ARMS17B:
                for s in S17.SEEDS17B[a]:
                    rows17b(fh, B.format(arm=a, s=s, meth="listed_sft", stage="sft"), L.expit(l0 + dl[a]), 8)
        for t in S17.STEPS17B:
            with open(runs / f"probes/prereg17b/{fam}_mf_s{t}.jsonl", "w") as fh:
                for a in ("grpo",) + S17.ARMS17B:
                    dd = dl[a] if a != "grpo" else np.zeros(n17)
                    for s in S17.SEEDS17B[a]:
                        rows17b(fh, B.format(arm=a, s=s, meth="full_grpo" if a == "grpo" else "listed_sft", stage=f"grpo/checkpoint-{t}"),
                                L.expit(l0 + 0.6 + dd * 0.2), 8)
    P(f"mock root written: {root}")


def dry_mock(args):
    root = Path(args.mock_root)
    make_mock(root)
    sys.path.insert(0, str(REPO))  # the frozen 17B loader imports route_sft_grpo from paths.root (the mock has none)
    import route_sft_grpo  # noqa: F401
    paths = S.Paths(str(root), "runs/gsm8k/Llama-3.2-1B-Instruct-px-{proto}{src}")
    args.B = min(args.B, 200)
    args.nsim = min(args.nsim, 40)
    P("\n# MOCK 17A unblind path (synthetic new arms; real Part A reference cells)")
    unblind_17a(paths, args)
    P("\n# MOCK 17B unblind path (synthetic, 120 items)")
    unblind_17b(paths, args)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["17A", "17B", "all"], default="all")
    ap.add_argument("--dry_run", choices=["synthetic", "aggregate", "mock"])
    ap.add_argument("--schema_only", action="store_true")
    ap.add_argument("--reference", choices=["partA"])
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--proto", choices=["mf", "cont"], default="mf")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--B", type=int, default=REG18["B"])
    ap.add_argument("--nsim", type=int, default=REG18["NSIM"])
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--sizes", default="17A,17B")
    ap.add_argument("--worlds", default=",".join(WORLDS))
    ap.add_argument("--steps17a", default="375")
    ap.add_argument("--calib_out", default=str(HERE / "p18_calib.jsonl"))
    ap.add_argument("--calib_in")
    ap.add_argument("--mock_root")
    args = ap.parse_args()
    P(f"# score_prereg18.py args={vars(args)}\n# REG18={REG18}\n# frozen libraries: " + ", ".join(
        f"{Path(m.__file__).name} {sha(m.__file__)}" for m in (L, S, S17)) + f"; this file {sha(__file__)}")
    t0 = time.time()
    paths = S.Paths(args.root, "runs/gsm8k/Llama-3.2-1B-Instruct-px-{proto}{src}")
    if args.dry_run == "synthetic":
        dry_synthetic(args)
    elif args.dry_run == "aggregate":
        recs = [json.loads(x) for x in open(args.calib_in) if x.strip()]
        aggregate(recs)
        grid(recs)
        minority_table(recs)
    elif args.dry_run == "mock":
        dry_mock(args)
    elif args.schema_only:
        schema18(paths, args.part, args.proto)
    elif args.reference == "partA":
        reference_partA(args)
    else:
        flags = glob.glob(str(REPO / "prereg" / "18_*.md"))
        if not args.unblind or not flags:
            sys.exit("refused: --unblind runs only after prereg 18 is registered (prereg/18_*.md must exist in the repository)")
        if Path(args.root).resolve() != REPO:
            sys.exit("refused: --unblind scores the repository root only")
        P(f"# registration flag: {flags}")
        if args.part in ("17A", "all"):
            unblind_17a(paths, args)
        if args.part in ("17B", "all"):
            unblind_17b(paths, args)
    P(f"\n# done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
