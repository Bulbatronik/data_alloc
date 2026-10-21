"""p15lib: helpers for score_prereg15.py (prereg 15 and 16). numpy only (the env has no scipy).

Contents
  graders    exact port of route_sft_grpo.py (extract_final_answer, normalize_number, numeric_answer,
             answer_is_correct) for the numeric datasets gsm8k, math, gsm_symbolic, gsm_plus, svamp
  loaders    probe.py outputs (keyed by example_id) and training-job predictions (row order = example_id)
  Cell       per-(set, arm, stage) correct counts by seed unit and by sample half (half 0 = samples < k//2)
  Design     cells + hierarchical bootstrap draws: seeds within lineage (one draw shared by every cell of a
             lineage, so an SFT adapter and its RL run move together), then items (optionally clustered)
  estimators EIV retention (G1 rho_item), cross-fitted Wald sign split + decomposition, EIV memory regression,
             base-success bins (R1_krl_checks)
  latent     2-D NPMLE prior (T_retro_e2_markov), tilt and memory links, simulation through identical cells
  part 1     partial Spearman (R2_rescore_checks2) and per-sample phi statistics
"""
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from types import SimpleNamespace

import numpy as np

# ============================================================== graders (port of route_sft_grpo.py l.104-160, 244-252)
_NUM = r"-?(?:\d[\d,]*)(?:\.\d+)?"
_ANSWER_PATTERNS = [
    rf"####\s*({_NUM})",  # the format the prompt asks for
    rf"\\boxed\{{\s*\\?\$?\s*({_NUM})",
    rf"final\s+answer(?:\s+is|:)?\s*\$?\s*({_NUM})",
]
_MARKED_PATTERNS = [
    rf"####\s*(?:<final numeric answer>)?[\s:]*\$?\s*\**\s*({_NUM})",
    rf"\\boxed\{{\s*\\?\$?\s*({_NUM})",
    rf"(?:final\s+)?answer\s*(?:is|:)\s*:?\s*\$?\s*\**\s*({_NUM})",
]


def numeric_answer(answer):
    """The answer if it is a plain number (MATH writes e.g. '10,\\!000'), else None."""
    if answer is None:
        return None
    cleaned = answer.replace("\\!", "").replace(",", "").replace("$", "").strip()
    return cleaned if re.fullmatch(r"-?\d+(\.\d+)?", cleaned) else None


def extract_final_answer(text, marked=False):
    text = text.split("\nQuestion:")[0]
    for pattern in _MARKED_PATTERNS if marked else _ANSWER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1]
    if marked:
        return None
    nums = re.findall(_NUM, text)
    return nums[-1] if nums else None


def normalize_number(x):
    if x is None:
        return None
    x = x.strip().replace(",", "").replace("$", "")
    x = x.rstrip(".")
    try:
        d = Decimal(x)
        out = format(d.normalize(), "f")
        if "." in out:
            out = out.rstrip("0").rstrip(".")
        return "0" if out in {"-0", ""} else out
    except InvalidOperation:
        return x


NUMERIC_DATASETS = {"gsm8k", "math", "gsm_symbolic", "gsm_plus", "svamp"}


def answer_is_correct(completion, gold, marked, dataset):
    """route_sft_grpo.answer_is_correct for the numeric datasets (geometry3k uses sympy and is not ported)."""
    if dataset not in NUMERIC_DATASETS:
        raise ValueError(f"no grader port for dataset {dataset!r}")
    return normalize_number(extract_final_answer(completion, marked)) == normalize_number(gold)


GRADERS = ("len", "mk")  # lenient = marked=False (primary), marked = marked=True (secondary)
OUTCOMES = ("len", "mk", "len_fin", "mk_fin", "stop")  # *_fin = correct AND stopped; stop = stopped


# ============================================================== loaders
def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class Eval:
    """One evaluation of one adapter on one item list: outcome arrays (N items x k samples), uint8."""

    def __init__(self, name, arrays, k, ids, gold):
        self.name, self.a, self.k, self.ids, self.gold = name, arrays, k, list(ids), list(gold)

    def select(self, ids):
        pos = {e: j for j, e in enumerate(self.ids)}
        j = np.array([pos[e] for e in ids])
        return Eval(self.name, {o: v[j] for o, v in self.a.items()}, self.k, ids, [self.gold[i] for i in j])


def _grade_row(r, dataset, stopped_key="stopped"):
    c = r["completion"]
    ln = answer_is_correct(c, r["gold"], False, dataset)
    mk = answer_is_correct(c, r["gold"], True, dataset)
    st = bool(r.get(stopped_key, True))
    return ln, mk, st, bool(r["correct"])


AGREE = Counter()  # marked regrade vs stored 'correct' (files whose stored field is marked)


def read_probe(paths, dataset, key_fn=lambda a: a, stored_is_marked=True):
    """probe.py outputs -> {key_fn(adapter): Eval}. Rows keyed by (adapter, example_id, sample); every adapter must
    have the same example_ids, each with samples 0..k-1 exactly once."""
    by = defaultdict(dict)  # adapter -> {(e, s): tuple}
    gold = defaultdict(dict)
    for p in ([paths] if isinstance(paths, str) else paths):
        for r in read_jsonl(p):
            a = r.get("adapter")
            key = (int(r["example_id"]), int(r["sample"]))
            if key in by[a]:
                raise ValueError(f"duplicate row {a} {key} in {p}")
            g = _grade_row(r, dataset)
            if stored_is_marked:
                AGREE[g[1] == g[3]] += 1
            by[a][key] = g
            gold[a][key[0]] = r["gold"]
    out = {}
    ref_ids = None
    for a, rows in by.items():
        ids = sorted({e for e, _ in rows})
        ks = Counter(e for e, _ in rows)
        k = max(ks.values())
        assert all(ks[e] == k for e in ids) and all((e, s) in rows for e in ids for s in range(k)), f"incomplete {a}"
        if ref_ids is None:
            ref_ids = ids
        assert ids == ref_ids, f"{a}: item set differs from the first adapter's"
        arr = {o: np.zeros((len(ids), k), np.uint8) for o in OUTCOMES}
        for j, e in enumerate(ids):
            for s in range(k):
                ln, mk, st, _ = rows[(e, s)]
                arr["len"][j, s], arr["mk"][j, s], arr["stop"][j, s] = ln, mk, st
                arr["len_fin"][j, s], arr["mk_fin"][j, s] = ln and st, mk and st
        name = key_fn(a)
        assert name not in out, f"two adapters map to {name}"
        out[name] = Eval(str(a), arr, k, ids, [gold[a][e] for e in ids])
    return out


def read_job(path, dataset, k=4, stored_is_marked=True):
    """Training-job predictions (route_sft_grpo.evaluate): row i is item i // k (= probe example_id under the same
    --data_seed), sample i % k."""
    rows = list(read_jsonl(path))
    assert len(rows) % k == 0, (path, len(rows))
    n = len(rows) // k
    arr = {o: np.zeros((n, k), np.uint8) for o in OUTCOMES}
    gold = []
    for i, r in enumerate(rows):
        assert r["sample"] == i % k, (path, i)
        ln, mk, st, sc = _grade_row(r, dataset)
        if stored_is_marked:
            AGREE[mk == sc] += 1
        j, s = divmod(i, k)
        arr["len"][j, s], arr["mk"][j, s], arr["stop"][j, s] = ln, mk, st
        arr["len_fin"][j, s], arr["mk_fin"][j, s] = ln and st, mk and st
        if s == 0:
            gold.append(r["gold"])
        else:
            assert r["gold"] == gold[-1]
    return Eval(path, arr, k, list(range(n)), gold)


def check_aligned(ev, ref_ids, ref_gold, label):
    """Assert a job Eval (row order) agrees with a probe item list on gold answers; return it restricted to ref_ids."""
    sub = ev.select(ref_ids)
    bad = sum(a != b for a, b in zip(sub.gold, ref_gold))
    assert bad == 0, f"{label}: {bad} gold mismatches vs probe example_ids"
    return sub


# --- schema-only (blinded data): counts, keys, adapter lists, gold alignment; never touches 'correct'
def schema_probe(paths):
    info = defaultdict(lambda: {"rows": 0, "items": set(), "samples": Counter(), "keys": set()})
    gold = defaultdict(dict)
    for p in ([paths] if isinstance(paths, str) else paths):
        for r in read_jsonl(p):
            a = r.get("adapter")
            d = info[a]
            d["rows"] += 1
            d["items"].add(int(r["example_id"]))
            d["samples"][int(r["sample"])] += 1
            d["keys"] |= set(r)
            gold[a][int(r["example_id"])] = r["gold"]
    return info, gold


def schema_job(path, k=4):
    rows = 0
    gold = []
    keys = set()
    for i, r in enumerate(read_jsonl(path)):
        assert r["sample"] == i % k
        keys |= set(r)
        if i % k == 0:
            gold.append(r["gold"])
        rows += 1
    return rows, gold, keys


# ============================================================== cells and design
class Cell:
    """Counts per outcome: C[o] shape (2 halves, U units, N items); Nh shape (2, U) samples per item per half.
    A unit is one seed (one SFT adapter / RL run); several evaluations of a unit are summed. lineage None = never
    resampled (the untrained model's evaluations are exchangeable samples of one model)."""

    def __init__(self, name, set_name, lineage, seeds, C, Nh):
        self.name, self.set, self.lineage, self.seeds, self.C, self.Nh = name, set_name, lineage, list(seeds), C, Nh
        self.U, self.N = Nh.shape[1], next(iter(C.values())).shape[2]

    @classmethod
    def from_evals(cls, name, set_name, lineage, units, outcomes=OUTCOMES):
        """units: list of (seed, [Eval, ...]) with identical item lists."""
        ids = units[0][1][0].ids
        U, N = len(units), len(ids)
        C = {o: np.zeros((2, U, N)) for o in outcomes}
        Nh = np.zeros((2, U))
        for u, (_, evs) in enumerate(units):
            for ev in evs:
                assert ev.ids == ids, f"{name}: item lists differ"
                h0 = np.arange(ev.k) < ev.k // 2
                for o in outcomes:
                    C[o][0, u] += ev.a[o][:, h0].sum(1)
                    C[o][1, u] += ev.a[o][:, ~h0].sum(1)
                Nh[0, u] += h0.sum()
                Nh[1, u] += (~h0).sum()
        c = cls(name, set_name, lineage, [s for s, _ in units], C, Nh)
        c.ids = ids
        return c

    def drop_seeds(self, bad):
        keep = [u for u, s in enumerate(self.seeds) if s not in bad]
        c = Cell(self.name, self.set, self.lineage, [self.seeds[u] for u in keep],
                 {o: v[:, keep] for o, v in self.C.items()}, self.Nh[:, keep])
        c.ids = getattr(self, "ids", None)
        return c


class Design:
    """strata: list of ('items', set, idx) (resample idx with replacement) or ('clusters', [ {set: idx}, ... ])
    (resample whole clusters). Default: every set's items resampled independently."""

    def __init__(self, cells, strata=None):
        self.cells = dict(cells)
        self.sets = {}
        for c in self.cells.values():
            assert self.sets.setdefault(c.set, c.N) == c.N, f"set {c.set}: item counts differ"
        self.lineages = {}
        for c in self.cells.values():
            if c.lineage is not None:
                self.lineages[c.lineage] = sorted(set(self.lineages.get(c.lineage, [])) | set(c.seeds))
        self.strata = strata or [("items", s, np.arange(n)) for s, n in self.sets.items()]

    def add(self, cell):
        self.cells[cell.name] = cell
        assert self.sets.setdefault(cell.set, cell.N) == cell.N
        if cell.lineage is not None:
            self.lineages[cell.lineage] = sorted(set(self.lineages.get(cell.lineage, [])) | set(cell.seeds))

    def draw(self, rng):
        wl = {L: Counter(rng.choice(s, len(s)).tolist()) for L, s in self.lineages.items()}
        J = defaultdict(list)
        for st in self.strata:
            if st[0] == "items":
                idx = st[2]
                J[st[1]].append(idx[rng.integers(0, len(idx), len(idx))])
            else:
                cl = st[1]
                for c in rng.integers(0, len(cl), len(cl)):
                    for s, idx in cl[c].items():
                        J[s].append(idx)
        return wl, {s: np.concatenate(v) for s, v in J.items()}


class View:
    """Per-item means of a design under one bootstrap draw (or the observed data when draw is None)."""

    def __init__(self, design, draw=None, rng=None):
        self.d = design
        self.wl, self.J = draw if draw is not None else (None, None)
        self.rng = rng
        self.cache = {}
        # item-only twin (same resampled items, seeds fixed): a reported comparison only (it under-covers the sign split)
        self.vi = self if self.wl is None else View(design, (None, self.J), rng)

    def idx(self, set_name):
        """Original item indices behind the (resampled) item axis of a set."""
        return np.arange(self.d.sets[set_name]) if self.J is None else self.J[set_name]

    def weights(self, cell):
        if self.wl is None or cell.lineage is None:
            return np.ones(cell.U)
        cnt = self.wl[cell.lineage]
        w = np.array([cnt.get(s, 0) for s in cell.seeds], float)
        if w.sum() == 0:  # an excluded seed was drawn for every slot: resample this cell's own units
            w = np.bincount(self.rng.integers(0, cell.U, cell.U), minlength=cell.U).astype(float)
        return w

    def m(self, name, o, scale="p"):
        """(half0, half1, full) per item; a half with no samples gives None. scale 'p': success proportions;
        scale 'l': logit((p * n + 0.5) / (n + 1)) with n the cell's nominal (unweighted) samples per item in that half
        or in total, so the transform is the same in every bootstrap replicate."""
        key = (name, o, scale)
        if key not in self.cache:
            if scale == "l":
                cell = self.d.cells[name]
                n0, n1 = cell.Nh.sum(1)
                h0, h1, f = self.m(name, o, "p")
                lg = lambda p, n: None if p is None else np.log((p * n + 0.5) / (n + 1)) - np.log(((1 - p) * n + 0.5) / (n + 1))
                self.cache[key] = (lg(h0, n0), lg(h1, n1), lg(f, n0 + n1))
                return self.cache[key]
            cell = self.d.cells[name]
            C = cell.C[o]
            if self.J is not None:
                C = C[:, :, self.J[cell.set]]
            w = self.weights(cell)
            num = np.einsum("u,hun->hn", w, C)
            den = cell.Nh @ w
            h0 = num[0] / den[0] if den[0] > 0 else None
            h1 = num[1] / den[1] if den[1] > 0 else None
            self.cache[key] = (h0, h1, num.sum(0) / den.sum())
        return self.cache[key]


def bootstrap(design, fn, B, seed, seeds=True):
    """fn(View) -> {key: float}. Returns (observed dict, {key: array of B replicates}). Each replicate resamples
    seeds within lineage (hierarchical; v) and items; v.vi is the same item draw with seeds fixed.
    seeds=False: item bootstrap only."""
    rng = np.random.default_rng(seed)
    full = fn(View(design))
    reps = defaultdict(list)
    for _ in range(B):
        wl, J = design.draw(rng)
        r = fn(View(design, (wl if seeds else None, J), rng))
        for k in full:
            reps[k].append(r.get(k, np.nan))
    return full, {k: np.array(v, float) for k, v in reps.items()}


def ci(a, q=(2.5, 97.5)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (np.nan, np.nan) if len(a) < 10 else tuple(np.percentile(a, q))


# ============================================================== estimators
# Every estimator takes a View, an outcome o, a quadruple (S, U, PS, PR) = (SFT start pre-RL, untrained pre-RL,
# SFT start post-RL, RL alone post-RL) and a scale ('p' probability, 'l' smoothed logit). Each returns additive
# "parts" (item-weighted sums), so pooling over arms or sets = summing parts, then dividing.
def cov(x, y):
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def pair_means(v, o, S, U, PS, PR, scale="p"):
    s0, s1, s = v.m(S, o, scale)
    u0, u1, u = v.m(U, o, scale)
    return s0, s1, s, u0, u1, u, v.m(PS, o, scale)[2], v.m(PR, o, scale)[2]


EIV_KEYS = ("num", "den", "bS", "bU", "n", "gap_pre", "gap_post", "rl_alone", "rl_sft")


def eiv_parts(v, o, S, U, PS, PR, mask=None, scale="p"):
    """b = Cov(D_post, Dbar) / Cov(D^1, D^2) (G1 rho_item; paper eq. bhat): numerator with all pre-RL samples
    (disjoint from post-RL samples), denominator from the two disjoint halves.
    Decomposition (paper eq. decomp, Prop. 7c), cross-fitted over the two half assignments:
    beta_pi = [Cov(q_pi - p_pi^1, D^2) + Cov(q_pi - p_pi^2, D^1)] / (2 Cov(D^1, D^2)); 1 + beta_S - beta_U equals b
    on the probability scale with equal halves. beta_U > 0: convergence (RL alone moves toward the SFT start);
    beta_S < 0: undoing (the SFT start moves against its own effect).
    mask: boolean over the set's ORIGINAL items (follows the bootstrap resampling)."""
    s0, s1, s, u0, u1, u, qs, qr = pair_means(v, o, S, U, PS, PR, scale)
    if mask is not None:
        m = mask[v.idx(v.d.cells[S].set)]
        s0, s1, s, u0, u1, u, qs, qr = (x[m] for x in (s0, s1, s, u0, u1, u, qs, qr))
    n = float(len(s))
    if n < 3:
        return {k: 0.0 for k in EIV_KEYS}
    D1, D2 = s0 - u0, s1 - u1
    return {"num": n * cov(qs - qr, s - u), "den": n * cov(D1, D2),
            "bS": n * 0.5 * (cov(qs - s0, D2) + cov(qs - s1, D1)),
            "bU": n * 0.5 * (cov(qr - u0, D2) + cov(qr - u1, D1)), "n": n,
            "gap_pre": float(np.sum(s - u)), "gap_post": float(np.sum(qs - qr)),
            "rl_alone": float(np.sum(qr - u)), "rl_sft": float(np.sum(qs - s))}


def eiv_from(P):
    n = P["n"]
    if n == 0 or P["den"] <= 0:
        return {k: np.nan for k in ("b", "beta_S", "beta_U", "VDpre", "gap_pre_pts", "gap_post_pts", "rl_alone_pts", "rl_sft_pts")}
    return {"b": P["num"] / P["den"], "beta_S": P["bS"] / P["den"], "beta_U": P["bU"] / P["den"], "VDpre": P["den"] / n,
            "gap_pre_pts": 100 * P["gap_pre"] / n, "gap_post_pts": 100 * P["gap_post"] / n,
            "rl_alone_pts": 100 * P["rl_alone"] / n, "rl_sft_pts": 100 * P["rl_sft"] / n}


WALD_KEYS = ("dpost", "dpre", "dS", "dU", "post0", "pre0", "n", "ok")


def wald_parts(v, o, S, U, PS, PR, tau, scale="p"):
    """Cross-fitted Wald (grouping) EIV estimator of sign-specific retention. In fold f the items are grouped by the
    PROBABILITY-scale half-f start difference D^f = p_S^f - p_U^f (helped '+': D^f >= tau; hurt '-': D^f <= -tau;
    reference '0': |D^f| < tau), and every quantity is MEASURED with the other half and the post-RL samples
    (independent of the selection: no regression to the mean, paper Prop. 7d).
    Primary b_side = sum_f [mean D_post(side) - mean D_post(0)] / sum_f [mean D_meas(side) - mean D_meas(0)]
    (chord slope from the reference group: robust to a common post-RL offset between the starts).
    Secondary 'origin' b_side = sum_f mean D_post(side) / sum_f mean D_meas(side) (paper Def. 1; through the origin).
    Chord decomposition: 1 - b_side = u_side + c_side, u = -chord(q_S - p_S^meas) / chord(D_meas) (undoing),
    c = chord(q_R - p_U^meas) / chord(D_meas) (convergence)."""
    ps0, ps1, _, pu0, pu1, _, _, _ = pair_means(v, o, S, U, PS, PR, "p")
    s0, s1, s, u0, u1, u, qs, qr = pair_means(v, o, S, U, PS, PR, scale)
    post = qs - qr
    out = {f"{sd}_{k}": 0.0 for sd in "pm" for k in WALD_KEYS}
    out["zero_n"] = 0.0
    for f in (0, 1):
        sel = (ps0 - pu0) if f == 0 else (ps1 - pu1)
        ms, mu = (s1, u1) if f == 0 else (s0, u0)
        meas, dS, dU = ms - mu, qs - ms, qr - mu
        plus, minus = sel >= tau - 1e-9, sel <= -tau + 1e-9
        zero = ~(plus | minus)
        out["zero_n"] += zero.sum() / 2
        for sd, mask in (("p", plus), ("m", minus)):
            if mask.sum() == 0 or zero.sum() == 0:
                continue
            out[f"{sd}_ok"] += 1
            out[f"{sd}_dpost"] += post[mask].mean() - post[zero].mean()
            out[f"{sd}_dpre"] += meas[mask].mean() - meas[zero].mean()
            out[f"{sd}_dS"] += dS[mask].mean() - dS[zero].mean()
            out[f"{sd}_dU"] += dU[mask].mean() - dU[zero].mean()
            out[f"{sd}_post0"] += post[mask].mean()
            out[f"{sd}_pre0"] += meas[mask].mean()
            out[f"{sd}_n"] += mask.sum() / 2
    return out


def wald_from(P, n_arms=1):
    r = {}
    for sd, lab in (("p", "plus"), ("m", "minus")):
        ok = P[f"{sd}_ok"] >= 2 * n_arms  # both folds of every arm had a non-empty side and reference group
        d = P[f"{sd}_dpre"]
        good = ok and abs(d) > 1e-12
        r[f"b_{lab}"] = P[f"{sd}_dpost"] / d if good else np.nan
        r[f"undo_{lab}"] = -P[f"{sd}_dS"] / d if good else np.nan
        r[f"conv_{lab}"] = P[f"{sd}_dU"] / d if good else np.nan
        r[f"origin_{lab}"] = P[f"{sd}_post0"] / P[f"{sd}_pre0"] if ok and abs(P[f"{sd}_pre0"]) > 1e-12 else np.nan
        r[f"dpre_{lab}_pts"] = 100 * d / (2 * n_arms)
        r[f"n_{lab}"] = P[f"{sd}_n"] / n_arms
    r["n_zero"] = P["zero_n"] / n_arms
    r["b_plus_minus_b_minus"] = r["b_plus"] - r["b_minus"]
    r["origin_plus_minus_origin_minus"] = r["origin_plus"] - r["origin_minus"]
    return r


def memory_parts(v, o, S, U, PS, PR, scale="p"):
    """Moments of the errors-in-variables linear projection of q on (p_own, p_other): true-score variances from the
    split halves, cross-covariances from disjoint sample sets (pooled = moments summed)."""
    s0, s1, s, u0, u1, u, qs, qr = pair_means(v, o, S, U, PS, PR, scale)
    n = float(len(s))
    return {"VS": n * cov(s0, s1), "VU": n * cov(u0, u1), "CSU": n * cov(s, u), "qs_s": n * cov(qs, s),
            "qs_u": n * cov(qs, u), "qr_u": n * cov(qr, u), "qr_s": n * cov(qr, s)}


def memory_from(P):
    """lam_S = coefficient of p_U in q_S ~ p_S + p_U; lam_R = coefficient of p_S in q_R ~ p_U + p_S (the placebo:
    RL alone never saw the SFT model). M = lam_S - lam_R: > 0 the SFT start's RL endpoint keeps the pre-SFT
    profile beyond its own (undoing toward the base); < 0 RL alone acquires the SFT profile beyond the base
    (convergence toward the SFT start)."""
    try:
        bS = np.linalg.solve([[P["VS"], P["CSU"]], [P["CSU"], P["VU"]]], [P["qs_s"], P["qs_u"]])
        bR = np.linalg.solve([[P["VU"], P["CSU"]], [P["CSU"], P["VS"]]], [P["qr_u"], P["qr_s"]])
    except (np.linalg.LinAlgError, KeyError):
        return {"own_S": np.nan, "lam_S": np.nan, "own_R": np.nan, "lam_R": np.nan, "M": np.nan}
    return {"own_S": bS[0], "lam_S": bS[1], "own_R": bR[0], "lam_R": bR[1], "M": bS[1] - bR[1]}


BIN_LABELS = ("0", "low", "mid", "high", "all")


def bin_index(frac):
    """0: no sample correct; low: (0, 1/3]; mid: (1/3, 2/3]; high: (2/3, 1); all: every sample correct."""
    e = 1e-9
    return np.where(frac <= e, 0, np.where(frac <= 1 / 3 + e, 1, np.where(frac <= 2 / 3 + e, 2,
                                                                         np.where(frac < 1 - e, 3, 4))))


def bin_parts(v, o, S, U, PS, PR, scale="p"):
    """R1_krl_checks (b): bins by the untrained start's PROBABILITY-scale success on half 0; D_pre = p_S (all) -
    p_U (half 1); D_post = q_S - q_R (all). Retention per bin = mean D_post / mean D_pre."""
    pu0 = v.m(U, o, "p")[0]
    s0, s1, s, u0, u1, u, qs, qr = pair_means(v, o, S, U, PS, PR, scale)
    lab = bin_index(pu0)
    d0, d1 = s - u1, qs - qr
    out = {}
    for k in range(5):
        m = lab == k
        out[f"bin{k}_n"], out[f"bin{k}_d0"], out[f"bin{k}_d1"] = float(m.sum()), float(d0[m].sum()), float(d1[m].sum())
    return out


def bins_from(P):
    r = {}
    for k, lab in enumerate(BIN_LABELS):
        n = P[f"bin{k}_n"]
        r[f"bin_{lab}_n"] = n
        r[f"bin_{lab}_D0"] = P[f"bin{k}_d0"] / n if n else np.nan
        r[f"bin_{lab}_D1"] = P[f"bin{k}_d1"] / n if n else np.nan
        r[f"bin_{lab}_ret"] = P[f"bin{k}_d1"] / P[f"bin{k}_d0"] if n and abs(P[f"bin{k}_d0"]) > 1e-12 else np.nan
    return r


def bins_used(obs, pre, min_pts=2.0, min_n=15):
    """Bins entering the base-success contrast: n >= 15 and |mean D_pre| >= 2 points (probability scale), fixed on
    the observed data. Returns (lowest-success label, highest-success label) or None."""
    ok = [lab for lab in BIN_LABELS if obs.get(pre + f"bin_{lab}_n", 0) >= min_n
          and abs(obs.get(pre + f"bin_{lab}_D0", 0)) * 100 >= min_pts]
    return (ok[0], ok[-1]) if len(ok) >= 2 else None


def add_parts(parts_list):
    out = defaultdict(float)
    for P in parts_list:
        for k, x in P.items():
            out[k] += x
    return dict(out)


SCALES = (("p", ""), ("l", "L_"))  # (scale, key prefix): probability scale keys bare, logit scale keys 'L_...'


def all_stats(v, o, quads, tau, prefix="", scales=SCALES):
    """Every registered statistic for a list of (S, U, PS, PR) quadruples pooled (parts summed over quadruples).
    All statistics use the hierarchical view v (seeds within lineage, then items): in the synthetic coverage check
    it covers b, b+ and b- at 0.97-1.00, whereas the item-only twin v.vi (seeds fixed) under-covers b+- (0.87-0.93).
    Memory is on the probability scale only (the smoothed-logit version is numerically unstable: the split-half
    moment matrix is near singular)."""
    vi = v  # registered: hierarchical for every statistic (synthetic coverage: item-only twin under-covers b+-, 0.87-0.93)
    r = {}
    for sc, kp in scales:
        E = add_parts([eiv_parts(v, o, *q, scale=sc) for q in quads])
        W = add_parts([wald_parts(vi, o, *q, tau, scale=sc) for q in quads])
        Bn = add_parts([bin_parts(vi, o, *q, scale=sc) for q in quads])
        parts = [eiv_from(E), wald_from(W, len(quads)), bins_from(Bn)]
        if sc == "p":
            parts.append(memory_from(add_parts([memory_parts(v, o, *q, scale=sc) for q in quads])))
        for part in parts:
            for k, x in part.items():
                if sc == "l" and (k.startswith("gap") or k.startswith("rl_") or k.startswith("n_") or k.endswith("_n")):
                    continue  # counts and point gaps are reported on the probability scale only
                r[prefix + kp + k] = x
    return r


# ============================================================== latent models (T_retro_e2 / T_retro_e2_markov)
def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


_LG = np.array([math.lgamma(i) for i in range(1, 2000)])  # _LG[n] = lgamma(n + 1) = log n!


def logC(n, c):
    n = np.asarray(n, int)
    c = np.asarray(c, int)
    return _LG[n] - _LG[c] - _LG[n - c]


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(7)
GH_W = GH_W / GH_W.sum()


def minimize(f, x0, maxiter=2000, xatol=1e-4, fatol=1e-4):
    """Nelder-Mead (standard coefficients), as in T_retro_e2.minimize."""
    x0 = np.asarray(x0, float)
    d = len(x0)
    S = [x0] + [x0 + np.eye(d)[i] * (0.25 if x0[i] == 0 else 0.25 * max(1.0, abs(x0[i]))) for i in range(d)]
    F = [f(s) for s in S]
    for _ in range(maxiter):
        o = np.argsort(F)
        S = [S[i] for i in o]
        F = [F[i] for i in o]
        if abs(F[-1] - F[0]) < fatol and max(np.abs(s - S[0]).max() for s in S[1:]) < xatol:
            break
        c = np.mean(S[:-1], 0)
        xr = c + (c - S[-1])
        fr = f(xr)
        if fr < F[0]:
            xe = c + 2 * (c - S[-1])
            fe = f(xe)
            S[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]:
            S[-1], F[-1] = xr, fr
        else:
            xc = c + 0.5 * (S[-1] - c)
            fc = f(xc)
            if fc < F[-1]:
                S[-1], F[-1] = xc, fc
            else:
                S = [S[0]] + [S[0] + 0.5 * (s - S[0]) for s in S[1:]]
                F = [F[0]] + [f(s) for s in S[1:]]
    i = int(np.argmin(F))
    return SimpleNamespace(x=S[i], fun=F[i])


class Latent2D:
    """NPMLE (EM on a fixed logit grid) of the joint prior of (logit p_U, logit p_S) from the pre-RL counts."""

    def __init__(self, cU, nU, cS, nS, grid=np.linspace(-6, 6, 17), iters=1500):
        A, B = np.meshgrid(grid, grid, indexing="ij")
        self.A, self.B = A.ravel(), B.ravel()
        lpA, lqA = np.log(expit(self.A)), np.log(expit(-self.A))
        lpB, lqB = np.log(expit(self.B)), np.log(expit(-self.B))
        cU, cS = np.asarray(cU, int), np.asarray(cS, int)
        LL = (logC(nU, cU)[:, None] + cU[:, None] * lpA + (nU - cU)[:, None] * lqA
              + logC(nS, cS)[:, None] + cS[:, None] * lpB + (nS - cS)[:, None] * lqB)
        LJ = np.exp(LL - LL.max(1, keepdims=True))
        g = np.full(len(self.A), 1 / len(self.A))
        for _ in range(iters):
            post = LJ * g
            post /= post.sum(1, keepdims=True)
            g = post.mean(0)
        self.g = g
        self.W = LJ * g
        self.Wsum = self.W.sum(1)

    def cond_nll(self, c1, n1, eta):
        """-log P(post counts | pre counts) for logit q = eta[grid atom, GH node] (shape (G, K) or (G, 1))."""
        lq, l1q = np.log(expit(eta)), np.log(expit(-eta))
        w = GH_W if eta.shape[1] == len(GH_W) else np.ones(1)
        c1 = np.asarray(c1, int)
        L1 = np.exp(logC(n1, c1)[:, None, None] + c1[:, None, None] * lq[None] + (n1 - c1)[:, None, None] * l1q[None])
        L1 = (L1 * w[None, None, :]).sum(2)
        return -float(np.sum(np.log((self.W * L1).sum(1) + 1e-300) - np.log(self.Wsum)))

    def draw(self, rng, n=None):
        """Per-item draws of (logit p_U, logit p_S) from each item's posterior given its own pre-RL counts
        (paper App. B step 2), so the simulated items keep the observed item composition."""
        post = self.W / self.Wsum[:, None]
        cdf = np.cumsum(post, 1)
        i = np.minimum((cdf < rng.random(len(post))[:, None] * cdf[:, -1:]).sum(1), len(self.A) - 1)
        return self.A[i], self.B[i]


def fit_tilt(lat, own, c1, n1):
    """One-index Gibbs tilt q = expit(logit p_own + s), s by ML (golden section on [-6, 6])."""
    f = lambda s: lat.cond_nll(c1, n1, (own + s)[:, None])
    lo, hi = -6.0, 6.0
    gr = (math.sqrt(5) - 1) / 2
    a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - gr * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + gr * (hi - lo)
            fb = f(b)
    return 0.5 * (lo + hi)


def fit_affine(lat, own, c1, n1):
    """One-index affine-logit link q = expit(gamma + beta * logit p_own) by ML (sensitivity null; paper Prop. 1c)."""
    r = minimize(lambda th: lat.cond_nll(c1, n1, (th[0] + th[1] * own)[:, None]), [1.0, 1.0])
    return {"gamma": r.x[0], "beta": r.x[1], "nll": r.fun}


def fit_link(lat, own, other, c1, n1, lam=None, x0=(0.8, 1.0, 0.0, 0.8)):
    """logit q = gamma + beta*own + lam*other + |tau|*z (T_retro_e2_markov). lam=None: free; else fixed."""
    def eta(th):
        lm = th[2] if lam is None else lam
        return th[0] + th[1] * own[:, None] + lm * other[:, None] + abs(th[-1]) * GH_X[None, :]
    if lam is None:
        r = minimize(lambda th: lat.cond_nll(c1, n1, eta(th)), list(x0))
        return {"gamma": r.x[0], "beta": r.x[1], "lam": r.x[2], "tau": abs(r.x[3]), "nll": r.fun}
    r = minimize(lambda th: lat.cond_nll(c1, n1, eta(th)), [x0[0], x0[1], x0[3]])
    return {"gamma": r.x[0], "beta": r.x[1], "lam": lam, "tau": abs(r.x[2]), "nll": r.fun}


def binom_cell(rng, cell, p, name=None, o="sim"):
    """A copy of cell's unit/half structure with Binomial(Nh, p_item) counts (no seed effects)."""
    C = np.zeros((2, cell.U, len(p)))
    for h in range(2):
        for u in range(cell.U):
            if cell.Nh[h, u] > 0:
                C[h, u] = rng.binomial(int(cell.Nh[h, u]), p)
    return Cell(name or cell.name, cell.set, cell.lineage, cell.seeds, {o: C}, cell.Nh.copy())


def counts(cell, o):
    """Total correct counts per item and the total samples per item (all units, both halves)."""
    return cell.C[o].sum((0, 1)).astype(int), int(cell.Nh.sum())


# ============================================================== part 1: partial Spearman and phi statistics
def rank(x):
    """Average ranks (ties share the mean rank), as in R2_rescore_checks2."""
    x = np.asarray(x, float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return (np.bincount(inv, weights=r) / cnt)[inv]


def pspear(y, x, Z):
    """Partial Spearman: Pearson correlation of rank residuals after OLS on [1, ranks of Z]; rows with any NaN
    dropped."""
    cols = [np.asarray(y, float), np.asarray(x, float)] + [np.asarray(z, float) for z in Z]
    ok = np.all([np.isfinite(c) for c in cols], axis=0)
    if ok.sum() < 20:
        return np.nan
    y, x = cols[0][ok], cols[1][ok]
    ry, rx = rank(y), rank(x)
    M = np.column_stack([np.ones(ok.sum())] + [rank(c[ok]) for c in cols[2:]])
    ey = ry - M @ np.linalg.lstsq(M, ry, rcond=None)[0]
    ex = rx - M @ np.linalg.lstsq(M, rx, rcond=None)[0]
    if ey.std() < 1e-12 or ex.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ey, ex)[0, 1])


def logit_sm(c, n):
    """logit((c + 0.5) / (n + 1)) (R2's smoothing)."""
    p = (np.asarray(c, float) + 0.5) / (n + 1)
    return np.log(p) - np.log1p(-p)
