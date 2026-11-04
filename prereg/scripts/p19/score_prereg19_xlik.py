#!/usr/bin/env python3
"""Score prereg/19 M2, "behavioural convergence": do the two starts end on the SAME per-item solutions, or only on the
same success rates? Written and dry-run BEFORE registration; no statistic of this file has been computed on real data.

Input: xlik.py outputs (teacher-forced log-likelihoods of every role's stored samples under every role of the arm:
U0 untrained, S0 SFT start, U1 RL alone at the final step, S1 SFT start at the final step; T = 1 primary), joined with
the samples' reward from the probe files, graded here with the frozen / pipeline graders:
  GSM8K S' runs: reward = marked-correct (the S' reward had no stop requirement);
  17B runs:      reward = marked-correct AND stopped (--reward marked --reward_needs_stop).

DEFINITIONS (per item i, per seed; a "reward-1 sample" is one the training reward would score 1):
  l_M(y) = log M(y) / nbar_i       sequence log-likelihood (completion tokens + the stop token if stopped) divided by an
                                   ITEM constant, nbar_i = mean token count of U0's samples of item i (a per-token scale
                                   that keeps the tilt identities below exact; per-sample per-token means reported)
  phi(y) = l_S0(y) - l_U0(y)       the pre-RL axis: > 0 = y looks like the SFT start rather than the base
  psi(y) = l_S1(y) - l_U1(y)       the post-RL axis
  E_X[f] = mean of f over role X's reward-1 samples of the item; items enter when all four roles have >= 1 (per seed)
  A_pre = E_S0[phi] - E_U0[phi]    (= 2 x the Jeffreys cross-likelihood gap of the pre-RL pair on its own correct samples)
  G_post = E_S1[psi] - E_U1[psi]   (the same gap for the post-RL pair)
  (i)  convergence  kappa = sum G_post / sum A_pre
  (i') axis retention b = sum (E_S1[phi] - E_U1[phi]) / sum A_pre = 1 - u - c, with
  (ii) undoing     u = - sum (E_S1[phi] - E_S0[phi]) / sum A_pre   (the SFT start's correct solutions move toward the base's)
       convergence c =   sum (E_U1[phi] - E_U0[phi]) / sum A_pre   (RL alone's correct solutions move toward the SFT start's)
       route share u_share = u / (u + c) (defined when u + c >= 0.2)
  Sums run over items and seeds (pooling over arms = summing).
NULL (exact, no fitting). Any reward tilt M1(y) = M0(y) w(r(y)) / Z — the KL-regularised optimum (Prop. 1) — leaves the
  distribution of a start's reward-1 samples unchanged, and adds the per-item constant log(q/p) to every reward-1
  sample's log-likelihood. Hence, at T = 1 and with an item-constant normalisation: E[kappa numerator] = A_pre, b = 1,
  u = c = 0. Samples were drawn at T = 0.6 (token-level temperature), where the invariance is only approximate: the
  synthetic dry run quantifies that deviation (worlds TILT at T = 1 and T = 0.6).
ALL-SAMPLE versions (reported): the same axis statistics over all samples, against the success-matched tilt comparator
  E^tilt_S1[phi] = q_S E_S0[phi | r=1] + (1 - q_S) E_S0[phi | r=0] (q_S = S1's observed reward rate on the item; an empty
  class falls back to the unconditional mean), likewise for RL alone.
SURFACE (reported): mean completion length per role.

Usage (CPU via Slurm):
  python -I runs/theory/scratch/score_prereg19_xlik.py --schema_only [--xlik_dir runs/probes/prereg19_xlik]
  python -I runs/theory/scratch/score_prereg19_xlik.py --dry_run synthetic [--reps 100 --procs 16]
  python -I runs/theory/scratch/score_prereg19_xlik.py --dry_run mock --mock_root <new dir>
  python -I runs/theory/scratch/score_prereg19_xlik.py --unblind [--xlik_dir ...]   # refused unless prereg/19_*.md exists
"""
import argparse
import glob
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FROZEN = REPO / "prereg" / "scripts" / "p17"
sys.path.insert(0, str(FROZEN))
import p15lib as L  # noqa: E402

NAN = float("nan")
ROLES = ("U0", "S0", "U1", "S1")
REG19X = dict(B=2000, MIN_MOVE=0.2, SHARE=0.65, KAPPA_MAX=0.6, KAPPA_CI=0.9, Z_GATE=8.0, TILT_LO=0.6, SEED=1931)
# Z_GATE: an arm is testable only if A_pre (the pre-RL separation) is >= Z_GATE bootstrap SEs from 0; TILT_LO: kappa CI
# lower bound > TILT_LO reads as "tilt-like" (no convergence beyond reweighting).)
G = "runs/gsm8k/Llama-3.2-1B-Instruct"
FAM = {  # mirrors xlik.FAMILIES (xlik.py imports torch; this scorer must not)
    "gsm8k": dict(dataset="gsm8k", k=4, files={"U0": "runs/probes/prereg14_a/untrained.jsonl", "S0": "runs/probes/prereg14_a/sft.jsonl",
                                               "U1": "runs/probes/prereg14_a/s375.jsonl", "S1": "runs/probes/prereg14_a/s375.jsonl"},
                  seeds=(42, 43, 44), arms=("gold_250", "teacher_250"), needs_stop=False),
    "llama17b": dict(dataset="math35d", k=8, fam="llama", seeds=(42, 43), arms=("gold", "teacher"), needs_stop=True),
    "gemma17b": dict(dataset="math35d", k=8, fam="gemma", seeds=(42, 43), arms=("gold", "teacher"), needs_stop=True),
}
for _f in ("llama17b", "gemma17b"):
    _p = FAM[_f]["fam"]
    FAM[_f]["files"] = {"U0": f"runs/probes/prereg17b/{_p}_mf_untrained.jsonl", "S0": f"runs/probes/prereg17b/{_p}_mf_sft.jsonl",
                        "U1": f"runs/probes/prereg17b/{_p}_mf_s300.jsonl", "S1": f"runs/probes/prereg17b/{_p}_mf_s300.jsonl"}
# proposed primary arms and their predicted route (from the already-scored arm-level routes, 15 / 17B)
ROUTE = {"gsm8k:gold_250": "undo", "gsm8k:teacher_250": "conv", "llama17b:gold": "undo", "llama17b:teacher": "conv",
         "gemma17b:gold": "undo", "gemma17b:teacher": "undo"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]


def fmt(x, ci=None, d=3):
    s = "nan" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"
    if ci is not None:
        s += " [nan]" if not np.isfinite(ci[0]) else f" [{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"
    return s


def P(*a):
    print(*a, flush=True)


# ================================================================ core: per-item parts from one seed's table
def item_parts(T, norm="item"):
    """T: {'r': {X: (N, k) 0/1 reward}, 'L': {M: {X: (N, k) log-lik incl. stop}}, 'n': {X: (N, k) tokens incl. stop}}.
    Returns {name: (N,) array} of additive per-item parts (NaN-free; 'inc' = inclusion indicator)."""
    r = {X: T["r"][X].astype(float) for X in ROLES}
    if norm == "item":
        nb = T["n"]["U0"].mean(1)[:, None]
        l = {M: {X: T["L"][M][X] / nb for X in ROLES} for M in ROLES}
    else:  # per-sample per-token means (sensitivity; the tilt identity is then only approximate)
        l = {M: {X: T["L"][M][X] / np.maximum(T["n"][X], 1) for X in ROLES} for M in ROLES}
    phi = {X: l["S0"][X] - l["U0"][X] for X in ROLES}
    psi = {X: l["S1"][X] - l["U1"][X] for X in ROLES}

    def cm(v, m):
        n = m.sum(1)
        return (v * m).sum(1) / np.maximum(n, 1), n > 0
    E, ok = {}, {}
    for X in ROLES:
        E[X], ok[X] = cm(phi[X], r[X])
    Ep = {X: cm(psi[X], r[X])[0] for X in ("S1", "U1")}
    inc = ok["U0"] & ok["S0"] & ok["U1"] & ok["S1"]
    z = lambda a: np.where(inc, a, 0.0)
    out = {"inc": inc.astype(float), "A_pre": z(E["S0"] - E["U0"]), "A_post": z(E["S1"] - E["U1"]),
           "dS": z(E["S1"] - E["S0"]), "dU": z(E["U1"] - E["U0"]), "G_post": z(Ep["S1"] - Ep["U1"])}
    # all samples + success-matched tilt comparator
    Ea = {X: phi[X].mean(1) for X in ROLES}

    def tilted(pre, q):
        m1, o1 = cm(phi[pre], r[pre])
        m0, o0 = cm(phi[pre], 1 - r[pre])
        m1 = np.where(o1, m1, Ea[pre])
        m0 = np.where(o0, m0, Ea[pre])
        return q * m1 + (1 - q) * m0
    Et = {"S1": tilted("S0", r["S1"].mean(1)), "U1": tilted("U0", r["U1"].mean(1))}
    out.update({"a_inc": np.ones(len(inc)), "a_pre": Ea["S0"] - Ea["U0"], "a_post": Ea["S1"] - Ea["U1"], "a_dS": Ea["S1"] - Ea["S0"],
                "a_dU": Ea["U1"] - Ea["U0"], "t_post": Et["S1"] - Et["U1"], "t_dS": Et["S1"] - Ea["S0"], "t_dU": Et["U1"] - Ea["U0"]})
    for X in ROLES:
        out[f"len_{X}"] = T["n"][X].mean(1)
        out[f"rew_{X}"] = r[X].mean(1)
    return out


def stats_from(S):
    """S: {part: summed value} -> statistics."""
    o = {}
    A = S["A_pre"]
    ok = abs(A) > 1e-12
    o["n_inc"] = S["inc"]
    o["A_pre_mean"] = A / S["inc"] if S["inc"] > 0 else NAN
    o["kappa"] = S["G_post"] / A if ok else NAN
    o["b"] = S["A_post"] / A if ok else NAN
    o["u"] = -S["dS"] / A if ok else NAN
    o["c"] = S["dU"] / A if ok else NAN
    mv = o["u"] + o["c"]
    o["u_share"] = o["u"] / mv if np.isfinite(mv) and mv >= REG19X["MIN_MOVE"] else NAN
    Aa = S["a_pre"]
    oka = abs(Aa) > 1e-12
    for k, num, sg in (("b_all", "a_post", 1), ("u_all", "a_dS", -1), ("c_all", "a_dU", 1),
                       ("b_tilt", "t_post", 1), ("u_tilt", "t_dS", -1), ("c_tilt", "t_dU", 1)):
        o[k] = sg * S[num] / Aa if oka else NAN
    for k in ("b", "u", "c"):
        o[f"{k}_all_ex"] = o[f"{k}_all"] - o[f"{k}_tilt"]
    for X in ROLES:
        o[f"len_{X}"] = S[f"len_{X}"] / S["a_inc"]
        o[f"rew_{X}"] = S[f"rew_{X}"] / S["a_inc"]
    return o


class ArmParts:
    """Per-arm parts: arrays (seeds, N) per part; items shared across arms of a family."""

    def __init__(self, name, per_seed):
        self.name = name
        self.seeds = sorted(per_seed)
        keys = per_seed[self.seeds[0]].keys()
        self.a = {k: np.stack([per_seed[s][k] for s in self.seeds]) for k in keys}
        self.N = self.a["inc"].shape[1]

    def sums(self, ws=None, ci=None):
        ws = np.ones(len(self.seeds)) if ws is None else ws
        ci = np.ones(self.N) if ci is None else ci
        return {k: float(ws @ v @ ci) for k, v in self.a.items()}


def set_stats(arms, draws=None):
    """arms: list of ArmParts (same item set); draws: (seed weights per arm, item counts) or None."""
    tot = defaultdict(float)
    for j, a in enumerate(arms):
        ws, ci = (None, None) if draws is None else (draws[0][j], draws[1])
        for k, x in a.sums(ws, ci).items():
            tot[k] += x
    return stats_from(tot)


def bootstrap(sets, B, seed):
    """sets: {name: [ArmParts]}; all arms of one call share the item set. Hierarchical: seeds within arm, then items."""
    rng = np.random.default_rng(seed)
    allarms = {a.name: a for v in sets.values() for a in v}
    N = next(iter(allarms.values())).N
    obs = {s: set_stats(v) for s, v in sets.items()}
    reps = {s: defaultdict(list) for s in sets}
    for _ in range(B):
        ci = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        wsd = {n: np.bincount(rng.integers(0, len(a.seeds), len(a.seeds)), minlength=len(a.seeds)).astype(float) for n, a in allarms.items()}
        for s, v in sets.items():
            st = set_stats(v, ([wsd[a.name] for a in v], ci))
            for k, x in st.items():
                reps[s][k].append(x)
    return obs, {s: {k: np.array(x) for k, x in d.items()} for s, d in reps.items()}


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (NAN, NAN) if len(a) < 10 else tuple(np.percentile(a, (2.5, 97.5)))


def rules(name, o, rp):
    """Proposed registered rules for one arm (route from ROUTE)."""
    route = ROUTE.get(name)
    k_lo, k_hi = ci(rp["kappa"])
    X1 = np.isfinite(o["kappa"]) and o["kappa"] <= REG19X["KAPPA_MAX"] and k_hi < REG19X["KAPPA_CI"]
    if route == "undo":
        lo = ci(rp["u"])[0]
        X2 = np.isfinite(o["u_share"]) and o["u_share"] >= REG19X["SHARE"] and lo > 0
    elif route == "conv":
        lo = ci(rp["c"])[0]
        X2 = np.isfinite(o["u_share"]) and 1 - o["u_share"] >= REG19X["SHARE"] and lo > 0
    else:
        X2 = None
    return X1, X2


def testable(o, rp):
    """Gate: the pre-RL separation A_pre is >= Z_GATE bootstrap SEs from 0; 'tilt-like' reading: kappa CI lower > TILT_LO."""
    se = float(np.nanstd(rp["A_pre_mean"]))
    z = o["A_pre_mean"] / se if se > 0 else NAN
    return float(z), bool(np.isfinite(z) and z >= REG19X["Z_GATE"]), bool(ci(rp["kappa"])[0] > REG19X["TILT_LO"])


def report(obs, reps, label):
    P(f"\n# {label}")
    keys = ("n_inc", "A_pre_mean", "kappa", "b", "u", "c", "u_share", "b_all", "b_tilt", "b_all_ex", "u_all", "u_tilt", "u_all_ex",
            "c_all", "c_tilt", "c_all_ex", "len_U0", "len_S0", "len_U1", "len_S1", "rew_U0", "rew_S0", "rew_U1", "rew_S1")
    for s in obs:
        P(f"\n  [{s}]")
        for k in keys:
            P(f"    {k:12s} {fmt(obs[s][k], ci(reps[s][k]))}")
        X1, X2 = rules(s, obs[s], reps[s])
        z, ok, tl = testable(obs[s], reps[s])
        P(f"    pre-RL separation: A_pre z = {z:.1f} (reported; z >= {REG19X['Z_GATE']} = well separated)")
        P(f"    PROPOSED X1 convergence (kappa <= {REG19X['KAPPA_MAX']} and CI upper < {REG19X['KAPPA_CI']}): "
          + ("HELD" if X1 else (f"REJECTED, tilt-like (kappa CI lower > {REG19X['TILT_LO']})" if tl else "INCONCLUSIVE"))
          + f"; X2 route ({ROUTE.get(s, '-')}; share >= {REG19X['SHARE']}, CI of the routed part > 0): "
          + ("-" if X2 is None else ("HELD" if X2 else "not held")))


# ================================================================ real data
def load_rewards(root, family, adapters, needs_stop):
    """{adapter: {(example_id, sample): reward-1}} graded with the pipeline's grader (marked; and stopped for 17B)."""
    F = FAM[family]
    want = {a for a in adapters}
    out = {a: {} for a in want}
    if F["dataset"] == "math35d":
        sys.path.insert(0, str(REPO))
        import route_sft_grpo as R
        grade = lambda c, g: R.answer_is_correct(c, g, True, "math35d")
    else:
        grade = lambda c, g: L.answer_is_correct(c, g, True, "gsm8k")
    files = {Path(root) / f for f in set(F["files"].values())}
    for f in files:
        for x in L.read_jsonl(f):
            a = x.get("adapter")
            if a not in want:
                continue
            ok = grade(x["completion"], x["gold"]) and (bool(x["stopped"]) or not needs_stop)
            out[a][(int(x["example_id"]), int(x["sample"]))] = int(ok)
    return out


def load_family(root, xlik_path, family, temp="lp1"):
    """xlik rows -> {arm: {seed: table}}; tables as item_parts expects. Duplicate rows (a resumed block) keep the last."""
    F = FAM[family]
    rows, ads = {}, {}
    for x in L.read_jsonl(xlik_path):
        rows[(x["seed"], x["scorer"], x["source"], x["example_id"], x["sample"])] = {
            "lp1": x["lp1"], "lp1_stop": x["lp1_stop"], "lp06": x["lp06"], "lp06_stop": x["lp06_stop"], "n_tok": x["n_tok"],
            "stopped": x["stopped"]}
        ads[(x["seed"], x["source"])] = x["source_adapter"]
    seeds = sorted({k[0] for k in rows})
    rew = load_rewards(root, family, {a for a in ads.values()}, F["needs_stop"])
    out = defaultdict(dict)
    for s in seeds:
        roles = sorted({k[2] for k in rows if k[0] == s})
        arms = sorted({r.split(":")[1] for r in roles if ":" in r})
        items = sorted({k[3] for k in rows if k[0] == s and k[2] == "U0"})
        pos = {e: i for i, e in enumerate(items)}
        k = F["k"]
        for arm in arms:
            rm = {"U0": "U0", "U1": "U1", "S0": f"S0:{arm}", "S1": f"S1:{arm}"}
            T = {"r": {}, "L": {M: {} for M in ROLES}, "n": {}}
            for X in ROLES:
                ad = ads[(s, rm[X])]
                T["r"][X] = np.zeros((len(items), k))
                T["n"][X] = np.zeros((len(items), k))
                for e in items:
                    for j in range(k):
                        T["r"][X][pos[e], j] = rew[ad][(e, j)]
                for M in ROLES:
                    A = np.full((len(items), k), np.nan)
                    for e in items:
                        for j in range(k):
                            x = rows[(s, rm[M], rm[X], e, j)]
                            A[pos[e], j] = x[temp] + (x[f"{temp}_stop"] or 0.0)
                            if M == "U0":
                                T["n"][X][pos[e], j] = x["n_tok"] + (1 if x["stopped"] else 0)
                    assert np.isfinite(A).all(), f"missing xlik rows {family} s{s} {M}<-{X}"
                    T["L"][M][X] = A
            out[arm][s] = T
    return out


def run_real(root, xlik_dir, B, label="REAL"):
    res = {}
    for family in FAM:
        f = Path(xlik_dir) / f"{family}.jsonl"
        if not f.exists():
            P(f"\n{family}: {f} missing; not scored")
            continue
        tabs = load_family(root, f, family)
        for norm in ("item", "sample"):
            sets = {f"{family}:{arm}": [ArmParts(f"{family}:{arm}", {s: item_parts(T, norm) for s, T in per.items()})]
                    for arm, per in tabs.items()}
            obs, reps = bootstrap(sets, B, REG19X["SEED"])
            report(obs, reps, f"{label} {family}, normalisation {norm} ({'PRIMARY' if norm == 'item' else 'sensitivity'}), T = 1")
            res[(family, norm)] = (obs, reps)
        tabs06 = load_family(root, f, family, temp="lp06")
        sets = {f"{family}:{arm}": [ArmParts(f"{family}:{arm}", {s: item_parts(T) for s, T in per.items()})] for arm, per in tabs06.items()}
        obs, reps = bootstrap(sets, min(B, 500), REG19X["SEED"])
        report(obs, reps, f"{label} {family}, likelihoods at T = 0.6 (reported)")
    return res


# ================================================================ synthetic worlds (class-level model of solutions)
SIZES = {"gsm8k": dict(N=1319, k=4, seeds=3, mu=0.0), "math": dict(N=1000, k=8, seeds=2, mu=-2.0)}
WORLDS = {  # name: (S1 = tilt of (1 - lam_S) S0 + lam_S U0, U1 = tilt of (1 - lam_U) U0 + lam_U S0)
    "TILT": (0.0, 0.0), "UNDO50": (0.5, 0.0), "UNDO80": (0.8, 0.0), "MATCH50": (0.0, 0.5), "MATCH80": (0.0, 0.8),
    "COMMON": (0.5, 0.5), "UNDO20": (0.2, 0.0), "MATCH20": (0.0, 0.2),
}
A_STRAT, NOISE = 4, 1.0


def _tilt(p, r, s):
    w = p * np.exp(s * r)
    return w / w.sum((-2, -1), keepdims=True)  # normalise over the joint (variant, correct/wrong)


def _sample(rng, p, k, T):
    """p: (N, 2A, 2) joint over (variant, correct/wrong). Token-level temperature on the two decisions (variant, then
    correct | variant). Returns (variant index, wrong indicator), each (N, k)."""
    pa = p.sum(-1)
    pc = p[..., 0] / np.maximum(pa, 1e-300)
    paT = pa ** (1 / T)
    paT /= paT.sum(-1, keepdims=True)
    pcT = pc ** (1 / T) / (pc ** (1 / T) + (1 - pc) ** (1 / T))
    N = p.shape[0]
    cdf = np.cumsum(paT, -1)
    a = np.minimum((rng.random((N, k))[..., None] > cdf[:, None, :]).sum(-1), p.shape[1] - 1)
    c = (rng.random((N, k)) < np.take_along_axis(pcT, a, 1)).astype(int)
    return a, 1 - c


def synth_multi(rng, size, arms, lamU=0.0, lamU_arm=None, T=0.6, sU=0.5, sS=1.0, noise=NOISE, style=0.9):
    """arms: {name: (lamS, damage)}. Class-level model of solutions per item: A strategies x 2 styles (base / SFT) x
    {correct, wrong}. U0 puts 0.9 of its mass on base-style variants; each arm's S0 (per seed) puts 0.9 on SFT-style
    variants, re-weights strategies (h) and shifts success by `damage` (logit). Post-RL: S1 = tilt((1 - lamS) S0 + lamS U0,
    sS), U1 = tilt((1 - lamU) U0 + lamU S0[lamU_arm], sU) (shared by the arms of a seed, as in the real runs).
    U0's samples are drawn ONCE and reused for every seed (the real untrained samples are one probe). Log-likelihoods
    are exact class log-probabilities plus N(0, noise^2) per (model, sample) (nats per sequence); style = the share of mass each start puts on its own style. Returns {arm: {seed: table}}."""
    Z = SIZES[size]
    N, k = Z["N"], Z["k"]
    g = rng.normal(0, 1.5, (N, A_STRAT))
    m = rng.normal(Z["mu"], 2.0, N)
    v = L.expit(m[:, None] + rng.normal(0, 1.0, (N, A_STRAT)))
    h = {a: rng.normal(0, 1.0, (N, A_STRAT)) for a in arms}
    lens = rng.uniform(150, 350, (N, 2 * A_STRAT))
    r = np.zeros((1, 1, 2))
    r[..., 0] = 1.0
    lv = np.log(v / (1 - v))

    def model(wst, gg, vv):
        pa = np.concatenate([np.exp(gg) * wst[0], np.exp(gg) * wst[1]], 1)
        pa /= pa.sum(1, keepdims=True)
        vc = np.concatenate([vv, vv], 1)
        return np.stack([pa * vc, pa * (1 - vc)], -1)
    U0 = model((style, 1 - style), g, v)
    sU0 = _sample(rng, U0, k, T)
    out = {a: {} for a in arms}
    for s in range(Z["seeds"]):
        S0 = {a: model((1 - style, style), g + h[a] + rng.normal(0, 0.3, (N, A_STRAT)), L.expit(lv + dmg + rng.normal(0, 0.2)))
              for a, (lamS, dmg) in arms.items()}
        S1 = {a: _tilt((1 - lamS) * S0[a] + lamS * U0, r, sS) for a, (lamS, _) in arms.items()}
        U1 = _tilt((1 - lamU) * U0 + lamU * S0[lamU_arm or next(iter(arms))], r, sU)
        sU1 = _sample(rng, U1, k, T)
        for a in arms:
            mods = {"U0": U0, "S0": S0[a], "U1": U1, "S1": S1[a]}
            smp = {"U0": sU0, "U1": sU1, "S0": _sample(rng, S0[a], k, T), "S1": _sample(rng, S1[a], k, T)}
            Tb = {"r": {}, "L": {M: {} for M in ROLES}, "n": {}}
            for X in ROLES:
                ai, cw = smp[X]
                Tb["r"][X] = 1 - cw
                Tb["n"][X] = np.take_along_axis(lens, ai, 1)
                for M in ROLES:
                    lp = np.log(np.maximum(mods[M][np.arange(N)[:, None], ai, cw], 1e-300))
                    Tb["L"][M][X] = lp + noise * rng.standard_normal((N, k))
            out[a][42 + s] = Tb
    return out


def calib_one(job):
    size, world, T, damage, rep, B, noise, style = job
    rng = np.random.default_rng(193000 + rep + 1000 * list(WORLDS).index(world) + 100000 * list(SIZES).index(size) + int(T * 10) * 7
                                + int(10 * damage + 50) * 13 + int(noise) * 17 + int(100 * style) * 19)
    lamS, lamU = WORLDS[world]
    per = synth_multi(rng, size, {"x": (lamS, damage)}, lamU=lamU, lamU_arm="x", T=T, noise=noise, style=style)["x"]
    arm = ArmParts("x", {s: item_parts(tb) for s, tb in per.items()})
    obs, reps = bootstrap({"x": [arm]}, B, rng.integers(1 << 31))
    o, rp = obs["x"], reps["x"]
    rec = {"size": size, "world": world, "T": T, "damage": damage, "rep": rep, "noise": noise, "style": style}
    for kk in ("kappa", "b", "u", "c", "u_share", "n_inc", "b_all_ex", "u_all_ex", "c_all_ex", "A_pre_mean"):
        rec[kk] = o[kk]
    rec["kappa_lo"], rec["kappa_hi"] = ci(rp["kappa"])
    rec["A_pre_z"], rec["testable"], rec["tiltlike"] = testable(o, rp)
    rec["u_lo"], rec["c_lo"] = ci(rp["u"])[0], ci(rp["c"])[0]
    return rec


def aggregate(recs):
    P("\n# M2 synthetic calibration (class-level solution model; reward-1 samples; item-constant normalisation)")
    by = defaultdict(list)
    for r in recs:
        by[(r["size"], r["damage"], r["T"], r.get("noise", 1.0), r.get("style", 0.9), r["world"])].append(r)
    for (size, dmg, T, nz, st) in sorted({k[:5] for k in by}):
        P(f"\n## size {size}, SFT success shift {dmg:+.1f} (logit), sampling T = {T}, noise {nz} nats/sequence, own-style share {st}")
        P("   world     n    n_inc  kappa med [10,90]        b      u      c      u_share   P(X1)  P(X2 undo)  P(X2 conv)  P(u_lo>0)  P(c_lo>0)"
          "  z med  P(gate)  P(gate&X1)  P(gate&X2u)  P(gate&X2c)  P(gate&tilt-like)")
        for w in WORLDS:
            rs = by.get((size, dmg, T, nz, st, w), [])
            if not rs:
                continue
            q = lambda kk: np.array([x.get(kk, np.nan) for x in rs], float)
            md = lambda kk: float(np.nanmedian(q(kk)))
            k10, k90 = np.nanpercentile(q("kappa"), [10, 90])
            X1 = (q("kappa") <= REG19X["KAPPA_MAX"]) & (q("kappa_hi") < REG19X["KAPPA_CI"])
            us = q("u_share")
            X2u = (np.nan_to_num(us, nan=-1) >= REG19X["SHARE"]) & (q("u_lo") > 0)
            X2c = (np.nan_to_num(1 - us, nan=-1) >= REG19X["SHARE"]) & (q("c_lo") > 0)
            g = q("testable") > 0.5
            tl = q("tiltlike") > 0.5
            P(f"   {w:8s} {len(rs):3d}  {md('n_inc'):6.0f}  {md('kappa'):+.3f} [{k10:+.3f},{k90:+.3f}]  {md('b'):+.3f} {md('u'):+.3f} "
              f"{md('c'):+.3f}  {md('u_share'):+.3f}    {X1.mean():.2f}    {X2u.mean():.2f}        {X2c.mean():.2f}        "
              f"{np.mean(q('u_lo') > 0):.2f}       {np.mean(q('c_lo') > 0):.2f}   {md('A_pre_z'):5.1f}  {g.mean():.2f}     "
              f"{(g & X1).mean():.2f}        {(g & X2u).mean():.2f}         {(g & X2c).mean():.2f}         {(g & tl).mean():.2f}")


def dry_synthetic(args):
    CONF = ((1.0, 1.0, 0.9), (0.6, 1.0, 0.9), (0.6, 10.0, 0.9), (0.6, 10.0, 0.6), (0.6, 30.0, 0.6))  # (T, noise, own-style share)
    jobs = [(size, w, T, dmg, rep, args.B, nz, st) for size in SIZES for dmg in (-0.8, 0.3) for (T, nz, st) in CONF for w in WORLDS
            for rep in range(args.reps)]
    P(f"# {len(jobs)} synthetic datasets, B {args.B}, {args.procs} processes")
    recs = []
    t0 = time.time()
    with Pool(args.procs) as pool, open(args.calib_out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(calib_one, jobs, chunksize=2)):
            f.write(json.dumps(r) + "\n")
            recs.append(r)
            if (i + 1) % 200 == 0:
                P(f"  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)")
    aggregate(recs)


# ================================================================ schema and mock
def schema(root, xlik_dir):
    P("# schema-only: xlik outputs (rows, blocks, seeds) and probe files' presence; no likelihood value is read")
    for family, F in FAM.items():
        for f in sorted(set(F["files"].values())):
            P(f"  {family} probe {f}: {'ok' if (Path(root) / f).exists() else 'MISSING'}")
        xf = Path(xlik_dir) / f"{family}.jsonl"
        if not xf.exists():
            P(f"  {family} xlik output {xf}: MISSING (the full xlik runs are not done)")
            continue
        blocks = defaultdict(int)
        keys = set()
        for x in L.read_jsonl(xf):
            blocks[(x["seed"], x["scorer"], x["source"])] += 1
            keys = keys or set(x)
        P(f"  {family} xlik output {xf.name}: {len(blocks)} (seed, scorer, source) blocks; rows per block "
          f"{sorted(set(blocks.values()))}; keys {sorted(keys)}")
        need = {(s, M, X) for s in F["seeds"] for a in F["arms"] for M in ("U0", "U1", f"S0:{a}", f"S1:{a}")
                for X in ("U0", "U1", f"S0:{a}", f"S1:{a}")}
        miss = sorted(need - set(blocks))
        P(f"    blocks needed for the proposed arms: {len(need)}; missing {len(miss)}" + (f" e.g. {miss[:4]}" if miss else ""))


def make_mock(root, seed=1937, N=40):
    """SYNTHETIC probe files and xlik output for gsm8k (gold_250 planted UNDO80, teacher_250 planted MATCH80)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    assert not (root / "runs").exists(), f"{root} has runs/ already: use a new directory"
    gold = [str(i + 3) for i in range(N)]
    probes = defaultdict(list)
    xrows = []
    roles_ad = lambda arm, s: {"U0": None, "U1": f"{G}-marked-steps375/seed_{s}/full_grpo/grpo/checkpoint-375",
                               "S0": f"{G}{'-qwen7b' if arm.startswith('teacher') else ''}/seed_{s}/random_sft_250/sft",
                               "S1": f"{G}-marked-steps375{'-qwen7b' if arm.startswith('teacher') else ''}/seed_{s}/random_sft_250/grpo/checkpoint-375"}
    seen_probe = set()
    SIZES["mock"] = dict(N=N, k=4, seeds=3, mu=0.0)
    per_arm = synth_multi(rng, "mock", {"gold_250": (0.8, -0.8), "teacher_250": (0.0, 0.3)}, lamU=0.8, lamU_arm="teacher_250")
    for arm, per in per_arm.items():
        for s, Tb in per.items():
            ad = roles_ad(arm, s)
            rn = {"U0": "U0", "U1": "U1", "S0": f"S0:{arm}", "S1": f"S1:{arm}"}
            for X in ROLES:
                if (ad[X], X) not in seen_probe:  # U0 / U1 shared by both arms: written once (first arm's draw)
                    seen_probe.add((ad[X], X))
                    f = FAM["gsm8k"]["files"][X]
                    for e in range(N):
                        for j in range(4):
                            c = Tb["r"][X][e, j]
                            probes[f].append({"adapter": ad[X], "example_id": e, "sample": j, "gold": gold[e], "stopped": True,
                                              "completion": f"#### {gold[e] if c else int(gold[e]) + 1}", "correct": bool(c),
                                              "prompt_style": "train"})
                for M in ROLES:
                    for e in range(N):
                        for j in range(4):
                            lp = float(Tb["L"][M][X][e, j])
                            xrows.append({"family": "gsm8k", "seed": s, "scorer": rn[M], "scorer_adapter": ad[M], "source": rn[X],
                                          "source_adapter": ad[X], "example_id": e, "sample": j, "stopped": True,
                                          "n_tok": int(Tb["n"][X][e, j]) - 1, "lp1": lp - 0.5, "lp1_stop": 0.5, "lp06": lp, "lp06_stop": 0.0})
    for f, rows in probes.items():
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    xd = root / "runs/probes/prereg19_xlik"
    xd.mkdir(parents=True, exist_ok=True)
    with open(xd / "gsm8k.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in xrows)
    P(f"mock written under {root}: gold_250 planted UNDO (lamS 0.8), teacher_250 planted MATCH (RL alone mixes 0.8 of the "
      "teacher start); U0 / U1 samples shared by the arms, as in the real runs")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", choices=["synthetic", "aggregate", "mock"])
    ap.add_argument("--schema_only", action="store_true")
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--xlik_dir", default=str(REPO / "runs/probes/prereg19_xlik"))
    ap.add_argument("--B", type=int, default=REG19X["B"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--calib_out", default=str(HERE / "p19_xlik_calib.jsonl"))
    ap.add_argument("--calib_in")
    ap.add_argument("--mock_root")
    args = ap.parse_args()
    P(f"# score_prereg19_xlik.py args={vars(args)}\n# REG19X={REG19X}\n# frozen p15lib {sha(L.__file__)}; this file {sha(__file__)}")
    t0 = time.time()
    if args.dry_run == "synthetic":
        dry_synthetic(args)
    elif args.dry_run == "aggregate":
        aggregate([json.loads(x) for x in open(args.calib_in) if x.strip()])
    elif args.dry_run == "mock":
        make_mock(args.mock_root)
        run_real(args.mock_root, Path(args.mock_root) / "runs/probes/prereg19_xlik", min(args.B, 300), label="MOCK (synthetic; planted "
                 "gold_250 UNDO80, teacher_250 MATCH80)")
    elif args.schema_only:
        schema(args.root, args.xlik_dir)
    else:
        flags = glob.glob(str(REPO / "prereg" / "19_*.md"))
        if not args.unblind or not flags:
            sys.exit("refused: --unblind runs only after prereg 19 is registered (prereg/19_*.md must exist in the repository)")
        if Path(args.root).resolve() != REPO:
            sys.exit("refused: --unblind scores the repository root only")
        run_real(args.root, args.xlik_dir, args.B)
    P(f"\n# done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
