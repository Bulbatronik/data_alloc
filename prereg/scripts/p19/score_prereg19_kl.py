#!/usr/bin/env python3
"""Score prereg 19 M5 (KL anchoring sweep, GSM8K) and M6 (Gemma dr_grpo, MATH L3-5). Written before any M5/M6 run.

Reuses the FROZEN prereg-17 copies of p15lib.py, score_prereg15.py and score_prereg17.py (prereg/scripts/p17/, imported
by path, unchanged): graders, Cells, the hierarchical bootstrap (seeds within lineage, then items), the EIV estimators
(b, beta_S, beta_U; c = beta_U / (1 - b) gated at 1 - b >= 0.15), and the one-index tilt null (NPMLE prior of
(logit p_U, logit p_S), ONE logit shift fitted on the untrained start's own pre/post pair, simulated through the
identical cells and estimators).

Usage (Slurm CPU only; never on the login node):
  python -I runs/theory/scratch/score_prereg19_kl.py --part {M5,M6} --schema_only
  python -I runs/theory/scratch/score_prereg19_kl.py --dry_run synthetic [--power_reps 60 --procs 4]
  python -I runs/theory/scratch/score_prereg19_kl.py --dry_run mock          # end-to-end on generated files
  python -I runs/theory/scratch/score_prereg19_kl.py --part {M5,M6} --unblind # refused unless prereg/19_*.md exists

M5 design (runs/theory/scratch/submit_prereg19_kl.sh). Arms gold-250, teacher-250 (qwen7b) and RL alone, merged-fresh,
beta in {0, 0.02, 0.1, 0.5}, seeds 42-44, checkpoints 100/200/300. Cells:
  U                 untrained: prereg-14 Part A probe (4 samples) + E2's three training-job evaluations (12): the
                    prereg-15 convention (same model, prompt, T 0.6, top-p 0.95, top-k 0, 512 tokens, data_seed 1234)
  <src>|pre@<b>     merged SFT start (probe --merged_start, runs/probes/prereg19_kl/start.jsonl), lineage <src>; the
                    same adapters start every beta, so pre and every beta's post cells are resampled jointly by seed
  <arm>@<b>|<t>     probe at checkpoint t (b<b>_s<t>.jsonl); <arm>@<b>|300job = the training job's own step-300
                    evaluation (predictions.jsonl, independent samples: replication set)
  KL:<arm>@<b>|<t>  per-item KL travelled from the run's own start (kl_travel.py, kl@0.6 by default)
Quad per (src, beta, t): (src|pre@b, U, src@b|t, grpo@b|t): the SFT start against RL alone AT THE SAME beta.
Exclusion (prereg 15): a run with GRPO stop rate < 0.5 over its last 20 steps (rollouts.jsonl) leaves that beta's cells
(its SFT adapter leaves that beta's pre cell); a cell with < 2 seeds is not scored.

M6 (submit_prereg19_drgrpo.sh): Gemma gold/teacher, seeds 42-43, dr_grpo, reusing the 17B SFT adapters, so the start
and untrained evaluations are 17B's (runs/probes/prereg17b/gemma_mf_{untrained,sft}.jsonl); post = probe k = 8 at
150/300 (runs/probes/prereg19_dr/gemma_dr_s{150,300}.jsonl); comparators = the 17B GRPO runs of the same adapters.
"""
import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "prereg/scripts/p17"
sys.path.insert(0, str(FROZEN))  # the frozen copies shadow the working copies next to this script
sys.path.insert(1, str(ROOT))    # route_sft_grpo (math35d grader, M6), after the frozen libraries
import p15lib as L  # noqa: E402
import score_prereg15 as S  # noqa: E402
import score_prereg17 as S17  # noqa: E402

P, fmt, med = S.P, S.fmt, S.med
GR = L.GRADERS
SEEDS = (42, 43, 44)
BETAS = (0.0, 0.02, 0.1, 0.5)
TAG = {0.0: "0", 0.02: "0.02", 0.1: "0.1", 0.5: "0.5"}
TAGS = tuple(TAG[b] for b in BETAS)
STEPS = (100, 200, 300)
SRCS = ("gold", "teacher")
ARMS = SRCS + ("grpo",)
PDIR = "runs/probes/prereg19_kl"
NAN = float("nan")

# Candidate thresholds (the lead registers them). Dry run (score_prereg19_dry_synthetic_out.txt, 60 datasets per world,
# GSM8K sizes, 3 seeds): the DIFF rule is the M5a primary (power 1.00; false positive 0.00 gold / 0.03 teacher in the
# flat world), RHO_BETA alone is secondary (false positive 0.07 / 0.22); for M5c only the step-stratified Spearman with
# its CI lower bound > 0 discriminates (false positive 0.00 / 0.03): the pooled 12-cell Spearman is >= 0.6 in EVERY world
# (it is driven by the step axis) and the step-stratified point rule >= 0.6 fires 0.35 / 0.45 in the flat world.
REG19 = dict(
    DIFF=0.15,         # M5a (primary): b(0.5, 300) - b(0, 300) >= 0.15 AND its 95% CI lower bound > 0, per source
    RHO_BETA=0.8,      # M5a (secondary): Spearman over the 4 betas of b(300) >= 0.8 (at most one adjacent swap)
    RHO_KL=0.6,        # M5c (reported point value; the candidate rule is the step-stratified CI lower bound > 0)
    STOP_MIN=0.5,      # exclusion (prereg 15)
    TRUNC_MARGIN=0.05,  # M6a: truncation share (steps 250-299) of a dr_grpo arm <= RL alone (17B GRPO) + 0.05
)


# ================================================================ M5: paths and loaders
def run_dir(root, arm, tag, seed):
    sfx = "-qwen7b" if arm == "teacher" else ""
    meth = "full_grpo" if arm == "grpo" else "random_sft_250"
    return Path(root) / f"runs/gsm8k/Llama-3.2-1B-Instruct-p19kl-b{tag}-mf{sfx}/seed_{seed}/{meth}"


_RX_POST = re.compile(r"/gsm8k/Llama-3\.2-1B-Instruct-p19kl-b([0-9.]+)-mf(-qwen7b)?/seed_(\d+)/(full_grpo|random_sft_250)"
                      r"/grpo/checkpoint-(\d+)/?$")
_RX_START = re.compile(r"/gsm8k/Llama-3\.2-1B-Instruct(-qwen7b)?/seed_(\d+)/random_sft_250/sft/?$")


def parse19(a):
    """adapter -> (arm, beta tag, seed, step | 'pre')."""
    m = _RX_POST.search("/" + a)
    if m:
        tag, teach, seed, meth, ck = m.groups()
        assert not (meth == "full_grpo" and teach), a
        return ("grpo" if meth == "full_grpo" else ("teacher" if teach else "gold"), tag, int(seed), int(ck))
    m = _RX_START.search("/" + a)
    assert m, f"unparsed adapter {a}"
    return ("teacher" if m.group(1) else "gold", None, int(m.group(2)), "pre")


def load_m5(root):
    root = Path(root)
    ev = {}
    files = sorted((root / PDIR).glob("b*_s*.jsonl")) + [root / PDIR / "start.jsonl"]
    for f in files:
        if f.exists():
            for k, e in L.read_probe(str(f), "gsm8k", key_fn=parse19).items():
                assert k not in ev, f"{k} evaluated twice"
                ev[k] = e
    unt = L.read_probe(str(root / "runs/probes/prereg14_a/untrained.jsonl"), "gsm8k")[None]
    e2u = [L.check_aligned(L.read_job(str(root / f"runs/gsm8k/Llama-3.2-1B-Instruct-censored-grpo/seed_{s}/base_predictions.jsonl"),
                                      "gsm8k"), unt.ids, unt.gold, f"E2 base seed {s}") for s in SEEDS]
    jobs = {}
    for arm in ARMS:
        for tag in TAGS:
            for s in SEEDS:
                d = run_dir(root, arm, tag, s)
                for name, stage in (("predictions.jsonl", "300job"), ("predictions_after_sft.jsonl", "prejob")):
                    f = d / name
                    if f.exists() and not (arm == "grpo" and stage == "prejob"):
                        jobs[(arm, tag, s, stage)] = L.check_aligned(L.read_job(str(f), "gsm8k"), unt.ids, unt.gold, f"job {arm} b{tag} {s} {name}")
    for k, e in ev.items():  # every probe evaluation covers the untrained evaluation's items with the same gold answers
        assert e.ids == unt.ids and e.gold == unt.gold, f"{k}: item list or gold answers differ from the untrained probe"
    return ev, unt, e2u, jobs


def load_kl(root, key):
    """{(arm, tag, step): {seed: (ids, k, array N x k of per-sample KL)}} from kl_b<tag>_s<t>.jsonl."""
    out = defaultdict(dict)
    for f in sorted((Path(root) / PDIR).glob("kl_b*_s*.jsonl")):
        by = defaultdict(dict)
        for r in L.read_jsonl(f):
            by[r["adapter"]][(int(r["example_id"]), int(r["sample"]))] = float(r[key])
        for a, rows in by.items():
            arm, tag, s, t = parse19(a)
            ids = sorted({e for e, _ in rows})
            k = max(j for _, j in rows) + 1
            assert all((e, j) in rows for e in ids for j in range(k)), f"incomplete KL rows for {a}"
            out[(arm, tag, t)][s] = (ids, k, np.array([[rows[(e, j)] for j in range(k)] for e in ids]))
    return out


def kl_cell(name, lineage, units, set_name):
    """A Cell whose outcome 'kl' holds per-half SUMS of per-sample KL, so View.m gives seed-weighted per-item means."""
    ids = units[0][1][0]
    C, Nh = np.zeros((2, len(units), len(ids))), np.zeros((2, len(units)))
    for u, (_, (ids_u, k, arr)) in enumerate(units):
        assert ids_u == ids, f"{name}: KL item lists differ across seeds"
        h0 = np.arange(k) < k // 2
        C[0, u], C[1, u] = arr[:, h0].sum(1), arr[:, ~h0].sum(1)
        Nh[0, u], Nh[1, u] = h0.sum(), (~h0).sum()
    c = L.Cell(name, set_name, lineage, [s for s, _ in units], {"kl": C}, Nh)
    c.ids = ids
    return c


def exclusions_m5(root):
    bad, rates = set(), {}
    for arm in ARMS:
        for tag in TAGS:
            for s in SEEDS:
                f = run_dir(root, arm, tag, s) / "grpo/rollouts.jsonl"
                if f.exists():
                    rates[(arm, tag, s)] = r = S.stop_rate_rollouts(str(f))[0]
                    if r < REG19["STOP_MIN"]:
                        bad.add((arm, tag, s))
    return bad, rates


def cells_m5(ev, unt, e2u, jobs, kl, bad, u_pool=True):
    """Per beta, a seed enters an arm's cells only if it is not excluded and has its pre-RL start (SFT arms) and all
    three probe checkpoints, so the pre and post cells of a quad hold the same seeds."""
    cells = {"U": L.Cell.from_evals("U", "gsm8k", None, [(0, [unt])] + ([(i + 1, [e]) for i, e in enumerate(e2u)] if u_pool else []))}
    full_ids = unt.ids
    missing = []
    for tag in TAGS:
        for arm in ARMS:
            ss = [s for s in SEEDS if (arm, tag, s) not in bad and all((arm, tag, s, t) in ev for t in STEPS)
                  and (arm == "grpo" or (arm, None, s, "pre") in ev)]
            missing += [(arm, tag, s) for s in SEEDS if s not in ss and (arm, tag, s) not in bad]
            if len(ss) < 2:
                continue
            lin = arm
            if arm != "grpo":
                cells[f"{arm}|pre@{tag}"] = L.Cell.from_evals(f"{arm}|pre@{tag}", "gsm8k", lin, [(s, [ev[(arm, None, s, "pre")]]) for s in ss])
            for t in STEPS:
                cells[f"{arm}@{tag}|{t}"] = L.Cell.from_evals(f"{arm}@{tag}|{t}", "gsm8k", lin, [(s, [ev[(arm, tag, s, t)]]) for s in ss])
                ku = [(s, kl[(arm, tag, t)][s]) for s in ss if s in kl.get((arm, tag, t), {})]
                if len(ku) >= 2:
                    set_name = "gsm8k" if ku[0][1][0] == full_ids else "gsm8k_kl"
                    cells[f"KL:{arm}@{tag}|{t}"] = kl_cell(f"KL:{arm}@{tag}|{t}", lin, ku, set_name)
            js = [s for s in ss if (arm, tag, s, "300job") in jobs]
            if len(js) >= 2:
                cells[f"{arm}@{tag}|300job"] = L.Cell.from_evals(f"{arm}@{tag}|300job", "gsm8k", lin, [(s, [jobs[(arm, tag, s, "300job")]]) for s in js])
    return cells, missing


# ================================================================ M5: statistics
def quad19(cells, src, tag, t):
    q = (f"{src}|pre@{tag}", "U", f"{src}@{tag}|{t}", f"grpo@{tag}|{t}")
    return q if all(c in cells for c in q) else None


def derived19(out, o, src):
    g = lambda k: out.get(k, NAN)
    b300 = [g(f"{o}|300|{src}@{tag}|b") for tag in TAGS]
    if np.all(np.isfinite(b300)):
        out[f"{o}|{src}|rho_beta_b300"] = S17.spearman(BETAS, b300)
        out[f"{o}|{src}|b300_hi_minus_lo"] = b300[-1] - b300[0]
    for lab, karm in (("S", src), ("R", "grpo")):
        er, kk, per_t = [], [], []
        for t in STEPS:
            et = [1 - g(f"{o}|{t}|{src}@{tag}|b") for tag in TAGS]
            kt = [g(f"kl|{t}|{karm}@{tag}") for tag in TAGS]
            er += et
            kk += kt
            per_t.append(S17.spearman(kt, et))
        if np.all(np.isfinite(er)) and np.all(np.isfinite(kk)):
            out[f"{o}|{src}|rho_erasure_kl{lab}"] = S17.spearman(kk, er)
            out[f"{o}|{src}|rho_erasure_kl{lab}_bystep"] = float(np.nanmean(per_t)) if np.any(np.isfinite(per_t)) else NAN
    # switch vs brake: erasure of the beta > 0 cells minus the beta = 0 trajectory's erasure at the SAME KL travelled by
    # the SFT start (piecewise linear through (0, 0) and the beta = 0 cells at t = 100, 200, 300; no extrapolation).
    # < 0: anchoring removes more erasure than the KL distance it saves would (switch); ~ 0: anchoring only brakes.
    x0 = [0.0] + [g(f"kl|{t}|{src}@0") for t in STEPS]
    y0 = [0.0] + [1 - g(f"{o}|{t}|{src}@0|b") for t in STEPS]
    if np.all(np.isfinite(x0 + y0)) and np.all(np.diff(x0) > 0):
        ex = [(1 - g(f"{o}|{t}|{src}@{tag}|b")) - float(np.interp(g(f"kl|{t}|{src}@{tag}"), x0, y0))
              for tag in TAGS[1:] for t in STEPS if np.isfinite(g(f"kl|{t}|{src}@{tag}")) and g(f"kl|{t}|{src}@{tag}") <= x0[-1]]
        ex = [x for x in ex if np.isfinite(x)]
        out[f"{o}|{src}|erasure_excess_matchedKL"] = float(np.mean(ex)) if ex else NAN
        out[f"{o}|{src}|n_matchedKL_cells"] = float(len(ex))


def stats19_fn(graders=GR, steps=STEPS + ("300job",)):
    def fn(v):
        cells, out = v.d.cells, {}
        for arm in ARMS:
            for tag in TAGS:
                for t in STEPS:
                    n = f"KL:{arm}@{tag}|{t}"
                    if n in cells:
                        out[f"kl|{t}|{arm}@{tag}"] = float(np.mean(v.m(n, "kl")[2]))
        for o in graders:
            for src in SRCS:
                for tag in TAGS:
                    for t in steps:
                        q = quad19(cells, src, tag, t)
                        if q:
                            out.update(S17.arm_stats(v, o, q, f"{o}|{t}|{src}@{tag}|"))
                derived19(out, o, src)
        return out
    return fn


def tilt_null(d, o, quad, nsim, seed):
    """S.run_nulls(..., kinds=('tilt',)) for one quad, without the memory/affine fits it does not use (identical tilt
    draws: run_nulls fits without the rng, then simulates with default_rng(seed))."""
    Sq, U, _, PR = quad
    cU, nU = L.counts(d.cells[U], o)
    cS, nS = L.counts(d.cells[Sq], o)
    cr, nr = L.counts(d.cells[PR], o)
    lat = L.Latent2D(cU, nU, cS, nS)
    sU = L.fit_tilt(lat, lat.A, cr, nr)
    sims = S.simulate_parts(np.random.default_rng(seed), d, quad, lat, S.generator({"sU": sU}, "tilt"), nsim, S.TAU)
    return S.derive_null([sims]), sU


def trl_kl(root, arm, tag, s, t):
    """Mean of TRL's logged 'kl' (T-0.8-tempered, per token, training pool) over log entries in (t - 100, t]."""
    f = run_dir(root, arm, tag, s) / f"grpo/checkpoint-{t}/trainer_state.json"
    if not f.exists():
        return NAN
    xs = [x["kl"] for x in json.load(open(f))["log_history"] if "kl" in x and t - 100 < x["step"] <= t]
    return float(np.mean(xs)) if xs else NAN


def report_m5(obs, reps, nulls, root):
    ci = lambda k: L.ci(reps[k]) if k in reps else (NAN, NAN)
    P("\n## M5 per-cell statistics (probability scale; L_ = smoothed logit; kl = mean per-item KL travelled, nats/sample)")
    for o in GR:
        for src in SRCS:
            for tag in TAGS:
                for t in STEPS + ("300job",):
                    pre = f"{o}|{t}|{src}@{tag}|"
                    if pre + "b" not in obs:
                        continue
                    kS, kR = obs.get(f"kl|{t}|{src}@{tag}", NAN), obs.get(f"kl|{t}|grpo@{tag}", NAN)
                    P(f"  [{o}] {src:7s} beta {tag:>4s} t {t!s:>6s}: b {fmt(obs[pre + 'b'], ci(pre + 'b'))}  beta_S {fmt(obs[pre + 'beta_S'])}"
                      f"  beta_U {fmt(obs[pre + 'beta_U'])}  c {fmt(obs.get(pre + 'c', NAN))}  L_b {fmt(obs.get(pre + 'L_b', NAN))}"
                      f"  VDpre {obs.get(pre + 'VDpre', NAN):.4f}  kl_S {kS:.3f}  kl_R {kR:.3f}")
    P("\n## TRL's logged KL (T 0.8, per token, training pool; mean of log entries in (t-100, t]; reported only)")
    for arm in ARMS:
        for tag in TAGS[1:]:
            P(f"  {arm:7s} beta {tag:>4s}: " + "  ".join(f"t {t}: " + ",".join(f"{trl_kl(root, arm, tag, s, t):.4f}" for s in SEEDS) for t in STEPS))
    res = {}
    for o in GR:
        P(f"\n## M5 candidate registered statistics [{o}] (thresholds REG19 are candidates: the lead registers them)")
        for src in SRCS:
            k = f"{o}|{src}|rho_beta_b300"
            bs = " ".join(f"{tag}: {obs.get(f'{o}|300|{src}@{tag}|b', NAN):+.3f}" for tag in TAGS)
            rho, dif = obs.get(k, NAN), obs.get(f"{o}|{src}|b300_hi_minus_lo", NAN)
            dlo, dhi = ci(f"{o}|{src}|b300_hi_minus_lo")
            P(f"  M5a {src}: b(300) by beta {bs}; b(0.5) - b(0) {fmt(dif, (dlo, dhi))} -> "
              + ("RISES (>= %.2f, CI > 0)" % REG19["DIFF"] if dif >= REG19["DIFF"] and dlo > 0 else "no rise shown")
              + f"; secondary: Spearman over beta {fmt(rho, ci(k))} -> "
              + ("monotone (>= %.1f)" % REG19["RHO_BETA"] if rho >= REG19["RHO_BETA"] else "not monotone"))
            nl = nulls.get((o, src, "0.5"))
            if nl is not None:
                lo, hi = np.percentile(nl[0]["b"][np.isfinite(nl[0]["b"])], [2.5, 97.5])
                b5 = obs.get(f"{o}|300|{src}@0.5|b", NAN)
                bl, bh = ci(f"{o}|300|{src}@0.5|b")
                P(f"  M5b {src}: b(0.5, 300) {fmt(b5, (bl, bh))} vs one-index null median {med(nl[0]['b']):+.3f}, 95% band "
                  f"[{lo:+.3f}, {hi:+.3f}] (s_U {nl[1]:+.3f}) -> " + ("INSIDE the null band" if lo <= b5 <= hi else "outside the null band")
                  + "; nulls at other betas: " + ", ".join(f"{tag}: {med(nulls[(o, src, tag)][0]['b']):+.3f}" for tag in TAGS if (o, src, tag) in nulls))
                res[(o, src, "M5b")] = bool(lo <= b5 <= hi)
            for lab, who in (("S", "SFT start"), ("R", "RL alone")):
                kp, ks = f"{o}|{src}|rho_erasure_kl{lab}", f"{o}|{src}|rho_erasure_kl{lab}_bystep"
                slo = ci(ks)[0]
                P(f"  M5c {src}: Spearman(1 - b, KL travelled by the {who}): step-stratified (mean over t of the Spearman over "
                  f"beta) {fmt(obs.get(ks, NAN), ci(ks))}" + ((" -> TRACKS (CI > 0)" if slo > 0 else " -> not shown") if lab == "S" else "")
                  + f"; pooled over 12 (beta, t) cells {fmt(obs.get(kp, NAN), ci(kp))} (uninformative: driven by the step axis)")
            kx = f"{o}|{src}|erasure_excess_matchedKL"
            xlo, xhi = ci(kx)
            P(f"  M5d {src} (reported): erasure of beta > 0 cells minus beta = 0 erasure at the same KL travelled "
              f"{fmt(obs.get(kx, NAN), (xlo, xhi))} over {obs.get(f'{o}|{src}|n_matchedKL_cells', 0):.0f} cells -> "
              + ("not computed" if not np.isfinite(xhi) else "switch beyond a brake (CI < 0)" if xhi < 0
                 else "consistent with a brake (CI includes 0)" if xlo <= 0 else "more erasure per KL (CI > 0)"))
            res[(o, src, "M5a")] = bool(dif >= REG19["DIFF"] and dlo > 0)
            res[(o, src, "M5c")] = bool(ci(f"{o}|{src}|rho_erasure_klS_bystep")[0] > 0)
    return res


def part_m5(root, args):
    P("\n# M5 — KL anchoring sweep (GSM8K, Llama-3.2-1B-Instruct, merged-fresh, beta in {0, 0.02, 0.1, 0.5})")
    ev, unt, e2u, jobs = load_m5(root)
    P(f"marked regrade vs stored 'correct': {L.AGREE[True]}/{sum(L.AGREE.values())} agree")
    bad, rates = exclusions_m5(root)
    P("GRPO stop rate, last 20 steps:", {f"{a}/b{t}/{s}": round(r, 3) for (a, t, s), r in sorted(rates.items())})
    P("excluded runs:", sorted(bad) or "none")
    kl = load_kl(root, args.kl_key)
    cells, missing = cells_m5(ev, unt, e2u, jobs, kl, bad)
    P("missing runs/evaluations (not excluded):", missing or "none")
    P("cells:", len(cells), sorted(c for c in cells if not c.startswith("KL:"))[:8], "...")
    d = L.Design(cells)
    t0 = time.time()
    obs, reps = L.bootstrap(d, stats19_fn(), args.B, seed=1901)
    P(f"bootstrap B = {args.B}: {time.time() - t0:.0f}s")
    nulls = {}
    for o in GR:
        for src in SRCS:
            for tag in TAGS:
                q = quad19(cells, src, tag, 300)
                if q:
                    nulls[(o, src, tag)] = tilt_null(d, o, q, args.nsim, seed=1902 + 10 * TAGS.index(tag) + (src == "teacher"))
    res = report_m5(obs, reps, nulls, root)
    du = L.Design(cells_m5(ev, unt, e2u, jobs, kl, bad, u_pool=False)[0])
    ou = stats19_fn(graders=("len",), steps=(300,))(L.View(du))
    P("  sensitivity, untrained = probe only (4 samples): " + "; ".join(
        f"{src} b(300) " + " ".join(f"{tag}: {ou.get(f'len|300|{src}@{tag}|b', NAN):+.3f}" for tag in TAGS) for src in SRCS))
    return res


# ================================================================ M6: Gemma dr_grpo
DR_ARMS = ("gold", "teacher")
DR_SEEDS = (42, 43)
DR_STEPS = (150, 300)
P19DR = "runs/probes/prereg19_dr"
_RX_DR = re.compile(r"/math35d/gemma-p19dr-mf-(gold|teacher|grpo)/seed_(\d+)/(listed_sft|full_grpo)/grpo/checkpoint-(\d+)/?$")


def dr_dir(root, arm, s, dr=True):
    meth = "full_grpo" if arm == "grpo" else "listed_sft"
    return Path(root) / f"runs/math35d/gemma-{'p19dr' if dr else 'p17b'}-mf-{arm}/seed_{s}/{meth}"


def load_dr(root):
    """dr_grpo checkpoint evaluations, regraded as score_prereg17.load_17b does (math35d sympy grader, lenient + marked)."""
    import route_sft_grpo as R
    evs, agree = {}, []
    for t in DR_STEPS:
        f = Path(root) / P19DR / f"gemma_dr_s{t}.jsonl"
        if not f.exists():
            continue
        by = defaultdict(dict)
        for r in L.read_jsonl(f):
            by[r.get("adapter")][(int(r["example_id"]), int(r["sample"]))] = r
        for a, rows in by.items():
            m = _RX_DR.search("/" + a)
            assert m, f"unparsed dr adapter {a}"
            arm, seed, _, ck = m.groups()
            ids = sorted({e for e, _ in rows})
            k = max(j for _, j in rows) + 1
            assert all((e, j) in rows for e in ids for j in range(k)), f"incomplete {a}"
            arr = {o_: np.zeros((len(ids), k), np.uint8) for o_ in L.OUTCOMES}
            for i, e in enumerate(ids):
                for j in range(k):
                    r = rows[(e, j)]
                    ln = R.answer_is_correct(r["completion"], r["gold"], False, "math35d")
                    mk = R.answer_is_correct(r["completion"], r["gold"], True, "math35d")
                    st = bool(r.get("stopped", True))
                    agree.append(mk == bool(r["correct"]))
                    arr["len"][i, j], arr["mk"][i, j], arr["stop"][i, j] = ln, mk, st
                    arr["len_fin"][i, j], arr["mk_fin"][i, j] = ln and st, mk and st
            evs[(arm + "_dr", int(seed), int(ck))] = L.Eval(str(a), arr, k, ids, [rows[(e, 0)]["gold"] for e in ids])
    return evs, (float(np.mean(agree)) if agree else NAN)


def trunc_share(path, lo=250, hi=299):
    """Share of GRPO rollouts that did not stop (hit the length limit) at logged steps lo..hi (resume replays removed)."""
    r = S.rollouts(str(path))
    xs = [not x["stopped"] for x in r if lo <= x["step"] <= hi]
    return float(np.mean(xs)) if xs else NAN


def stats_m6_fn(graders=GR):
    def fn(v):
        cells, out = v.d.cells, {}
        acc = lambda n, o: 100 * float(np.mean(v.m(n, o)[2]))
        for o in graders:
            for arm in DR_ARMS:
                if f"{arm}|pre" not in cells:
                    continue
                pre = acc(f"{arm}|pre", o)
                out[f"{o}|pre|{arm}|acc"] = pre
                for t in DR_STEPS:
                    for lab, n in (("grpo", f"{arm}|{t}"), ("dr", f"{arm}_dr|{t}")):
                        if n in cells:
                            out[f"{o}|{t}|{arm}|acc_{lab}"] = acc(n, o)
                            out[f"{o}|{t}|{arm}|gain_{lab}"] = acc(n, o) - pre
                    if f"{o}|{t}|{arm}|gain_dr" in out and f"{o}|{t}|{arm}|gain_grpo" in out:
                        out[f"{o}|{t}|{arm}|gain_dr_minus_grpo"] = out[f"{o}|{t}|{arm}|gain_dr"] - out[f"{o}|{t}|{arm}|gain_grpo"]
                    for lab, n in (("grpo", f"{arm}|{t}"), ("dr", f"{arm}_dr|{t}")):
                        q = (f"{arm}|pre", "U", n, f"grpo|{t}")
                        if all(c in cells for c in q):
                            st = S17.arm_stats(v, o, q, "")
                            for kk in ("b", "beta_S", "beta_U", "c"):
                                out[f"{o}|{t}|{arm}|{kk}_{lab}"] = st[kk]
        for n in cells:
            if "|" in n and n.split("|")[1] in ("150", "300"):
                out[f"evaltrunc|{n}"] = 100 * (1 - float(np.mean(v.m(n, "stop")[2])))
        return out
    return fn


def part_m6(root, args):
    P("\n# M6 — Gemma-3-1b-it, MATH L3-5 (math35d), merged-fresh, dr_grpo vs the 17B GRPO runs of the same SFT adapters")
    root = Path(root)
    import route_sft_grpo  # noqa: F401  (cached for score_prereg17.load_17b, which imports it from its root)
    ev17, ag17 = S17.load_17b(SimpleNamespace(root=root), "gemma", "mf")
    evdr, agdr = load_dr(root)
    P(f"marked regrade vs stored 'correct': 17B {ag17:.5f}, dr {agdr:.5f}")
    tr = {}
    for arm in DR_ARMS + ("grpo",):
        for s in SEEDS:
            for dr in (False, True):
                f = dr_dir(root, arm, s, dr) / "grpo/rollouts.jsonl"
                if f.exists():
                    tr[(arm, dr, s)] = (trunc_share(f), S.stop_rate_rollouts(str(f))[0])
    P("\n## M6a training truncation share, logged steps 250-299 (stop rate over the last 20 steps in brackets)")
    for (arm, dr, s), (x, st) in sorted(tr.items()):
        P(f"  {arm:7s} {'dr_grpo' if dr else 'grpo (17B)':10s} seed {s}: {x:.3f} [{st:.3f}]")
    rl = [tr[("grpo", False, s)][0] for s in SEEDS if ("grpo", False, s) in tr]
    rl_level = float(np.mean(rl)) if rl else NAN
    res = {}
    for arm in DR_ARMS:
        x = [tr[(arm, True, s)][0] for s in DR_SEEDS if (arm, True, s) in tr]
        g = [tr[(arm, False, s)][0] for s in DR_SEEDS if (arm, False, s) in tr]
        if x:
            ok = float(np.mean(x)) <= rl_level + REG19["TRUNC_MARGIN"]
            P(f"  M6a {arm}: dr_grpo {np.mean(x):.3f} (17B GRPO {np.mean(g) if g else NAN:.3f}) vs RL alone (17B GRPO) {rl_level:.3f} + "
              f"{REG19['TRUNC_MARGIN']} -> " + ("no blow-up (HELD)" if ok else "blow-up persists (not held)"))
            res[(arm, "M6a")] = ok
    for excl in (False, True):
        bad = {(a, dr, s) for (a, dr, s), (_, st) in tr.items() if st < REG19["STOP_MIN"]} if excl else set()
        cells = {"U": L.Cell.from_evals("U", "math35d", None, [(0, [ev17[("U", 0, "pre")]])])}
        for arm in DR_ARMS:
            ss = [s for s in DR_SEEDS if (arm, s, "pre") in ev17]
            if len(ss) >= 2:
                cells[f"{arm}|pre"] = L.Cell.from_evals(f"{arm}|pre", "math35d", arm, [(s, [ev17[(arm, s, "pre")]]) for s in ss])
            for t in DR_STEPS:
                for key, name, drf in ((arm, f"{arm}|{t}", False), (arm + "_dr", f"{arm}_dr|{t}", True)):
                    src = evdr if drf else ev17
                    uu = [(s, [src[(key, s, t)]]) for s in DR_SEEDS if (key, s, t) in src and (arm, drf, s) not in bad]
                    if len(uu) >= 2:
                        cells[name] = L.Cell.from_evals(name, "math35d", arm, uu)
        for t in DR_STEPS:
            uu = [(s, [ev17[("grpo", s, t)]]) for s in SEEDS if ("grpo", s, t) in ev17 and ("grpo", False, s) not in bad]
            if len(uu) >= 2:
                cells[f"grpo|{t}"] = L.Cell.from_evals(f"grpo|{t}", "math35d", "grpo", uu)
        d = L.Design(cells)
        obs, reps = L.bootstrap(d, stats_m6_fn(), args.B, seed=1911)
        ci = lambda k: L.ci(reps[k]) if k in reps else (NAN, NAN)
        P(f"\n## M6b mean accuracy (points; probe k = 8, 1,000 items) — {'17B stop-rate exclusion applied' if excl else 'all runs (primary candidate)'}")
        for o in GR:
            for arm in DR_ARMS:
                for t in DR_STEPS:
                    k = f"{o}|{t}|{arm}|gain_dr_minus_grpo"
                    if k not in obs:
                        P(f"  [{o}] {arm} t {t}: not scored (cells missing)")
                        continue
                    lo, hi = ci(k)
                    P(f"  [{o}] {arm} t {t}: pre {obs[f'{o}|pre|{arm}|acc']:.1f}; GRPO {obs[f'{o}|{t}|{arm}|acc_grpo']:.1f} "
                      f"(gain {fmt(obs[f'{o}|{t}|{arm}|gain_grpo'], ci(f'{o}|{t}|{arm}|gain_grpo'), 1)}); dr_grpo {obs[f'{o}|{t}|{arm}|acc_dr']:.1f} "
                      f"(gain {fmt(obs[f'{o}|{t}|{arm}|gain_dr'], ci(f'{o}|{t}|{arm}|gain_dr'), 1)}); dr - GRPO {fmt(obs[k], (lo, hi), 1)}"
                      + (" -> faster repair (HELD)" if t == 300 and obs[k] > 0 and lo > 0 else "")
                      + f"; b grpo {obs.get(f'{o}|{t}|{arm}|b_grpo', NAN):+.3f} dr {obs.get(f'{o}|{t}|{arm}|b_dr', NAN):+.3f}")
                    if t == 300 and not excl:
                        res[(o, arm, "M6b")] = bool(obs[k] > 0 and lo > 0)
        P("  evaluation truncation (% not stopped at 1,024 tokens): " + ", ".join(
            f"{k.split('|', 1)[1]} {v:.1f}" for k, v in sorted(obs.items()) if k.startswith("evaltrunc|")))
    return res


# ================================================================ schema-only (no outcome field is read)
def schema19(root, part):
    root = Path(root)
    P(f"# schema-only check, part {part}: file presence, counts, adapter parsing; no outcome field is read")
    if part == "M5":
        for arm in ARMS:
            for tag in TAGS:
                for s in SEEDS:
                    d = run_dir(root, arm, tag, s)
                    fs = [f"grpo/checkpoint-{t}/adapter_model.safetensors" for t in STEPS] + ["grpo/rollouts.jsonl", "grpo/merged_start.json", "predictions.jsonl"]
                    P(f"  {arm:7s} b{tag:<4s} seed {s}: " + ", ".join(f"{f.split('/')[-2] if 'checkpoint' in f else Path(f).name} {'ok' if (d / f).exists() else 'MISSING'}" for f in fs))
        for f in [root / PDIR / "start.jsonl"] + sorted((root / PDIR).glob("b*_s*.jsonl")) + sorted((root / PDIR).glob("kl_*.jsonl")):
            if not f.exists():
                P(f"  {f.relative_to(root)}: MISSING")
                continue
            info, _ = L.schema_probe(str(f)) if not f.name.startswith("kl_") else (None, None)
            if info is None:
                n, ads = 0, set()
                for r in L.read_jsonl(f):
                    n += 1
                    ads.add(r["adapter"])
                P(f"  {f.name}: {len(ads)} adapters, {n} rows, parsed {sorted({parse19(a)[:2] for a in ads}, key=str)}")
            else:
                P(f"  {f.name}: {len(info)} adapters, rows {sorted({d_['rows'] for d_ in info.values()})}, items "
                  f"{sorted({len(d_['items']) for d_ in info.values()})}, parsed {sorted({parse19(a)[::3] for a in info}, key=str)}")
        for f in ("runs/probes/prereg14_a/untrained.jsonl",) + tuple(f"runs/gsm8k/Llama-3.2-1B-Instruct-censored-grpo/seed_{s}/base_predictions.jsonl" for s in SEEDS):
            P(f"  {f}: {'exists' if (root / f).exists() else 'MISSING'}")
    else:
        for arm in DR_ARMS:
            for s in DR_SEEDS:
                d = dr_dir(root, arm, s)
                fs = [f"grpo/checkpoint-{t}/adapter_model.safetensors" for t in DR_STEPS] + ["grpo/rollouts.jsonl", "grpo/merged_start.json"]
                P(f"  dr {arm} seed {s}: " + ", ".join(f"{f} {'ok' if (d / f).exists() else 'MISSING'}" for f in fs))
                ms = d / "grpo/merged_start.json"
                if ms.exists():
                    P(f"     merged start: {json.load(open(ms))['sft_adapter']}")
        for t in DR_STEPS:
            f = root / P19DR / f"gemma_dr_s{t}.jsonl"
            if f.exists():
                info, _ = L.schema_probe(str(f))
                P(f"  {f.name}: {len(info)} adapters, rows {sorted({d_['rows'] for d_ in info.values()})}")
            else:
                P(f"  {f.relative_to(root)}: MISSING")
        for tag in ("untrained", "sft", "s150", "s300"):
            f = root / f"runs/probes/prereg17b/gemma_mf_{tag}.jsonl"
            P(f"  {f.relative_to(root)}: {'exists' if f.exists() else 'MISSING'}")


# ================================================================ synthetic worlds (dry runs)
WORLDS = {
    "switch": "retention rises with beta: KL anchoring suppresses erasure (rho(beta) = 1 / (1 + beta / 0.04)) faster than "
              "it slows reward progress (a_R(beta) = 1 / (1 + beta / 0.1))",
    "none": "retention flat in beta: anchoring slows reward progress (a_R) but not erasure (rho = 1)",
    "slowdown": "erasure slowed exactly as much as reward progress (rho = a_R): b rises with beta, erasure per unit KL fixed",
}


def kl_latent(rng, world, N=1319, seeds=SEEDS, kappa=0.0066, kl_scale=5.0):
    """Per-item success probabilities and mean KL travelled for every (cell, seed), on the logit scale of
    score_prereg17.ar_world (theta, psi, gold/teacher specs, seed effects 0.15; single rate on cumulative lr):
      RL alone   l_R(t, b) = l_U + w(t) a_R(b) (m + psi + g)
      SFT start  l_S(t, b) = l_U + u + d + w(t) a_R(b) (m + psi + g') - w(t) rho(b) d,   d = delta_r + h (reachable)
    KL travelled (proxy) = kl_scale * (l(t) - l(0))^2 per item."""
    th = rng.normal(-0.15, 2.3, N)
    lU = th + rng.normal(0, 0.8, N)
    psi, m = rng.normal(0, 1.2, N), 1.0
    aR = lambda b: 1 / (1 + b / 0.1)
    rho = {"switch": lambda b: 1 / (1 + b / 0.04), "none": lambda b: 1.0, "slowdown": aR}[world]
    w = lambda t: 1 - np.exp(-kappa * S17.tau_const(t))
    lat = {"U": L.expit(lU)}
    for b in BETAS:
        tag = TAG[b]
        for s in seeds:
            g = rng.normal(0, 0.15, N)
            for t in STEPS:
                lr = lU + w(t) * aR(b) * (m + psi + g)
                lat[("grpo", tag, t, s)] = L.expit(lr)
                lat[("KL", "grpo", tag, t, s)] = kl_scale * (lr - lU) ** 2
    spec = {"gold": dict(c=0.2, sd=1.3, mu_r=-0.7), "teacher": dict(c=0.8, sd=0.9, mu_r=0.0, mu_u=0.16, sd_u=0.6)}
    for src, sp in spec.items():
        a = sp["c"] * sp["sd"] ** 2 / 1.2 ** 2
        sr = math.sqrt(max(sp["sd"] ** 2 - a ** 2 * 1.2 ** 2, 1e-6))
        dr = a * psi + rng.normal(sp.get("mu_r", 0.0), sr, N)
        u = rng.normal(sp.get("mu_u", 0.0), sp.get("sd_u", 0.0), N)
        for s in seeds:
            d = dr + rng.normal(0, 0.15, N)
            l0 = lU + u + d
            lat[(src, None, "pre", s)] = L.expit(l0)
            for b in BETAS:
                tag = TAG[b]
                g = rng.normal(0, 0.15, N)
                for t in STEPS:
                    ls = l0 + w(t) * aR(b) * (m + psi + g) - w(t) * rho(b) * d
                    lat[(src, tag, t, s)] = L.expit(ls)
                    lat[("KL", src, tag, t, s)] = kl_scale * (ls - l0) ** 2
    return lat


def _bin_cell(rng, name, lineage, probs, k, seeds, set_name="gsm8k"):
    return S17._units(rng, name, lineage, probs, k, set_name, seeds)


def _kl_draw(rng, mean, k):
    return rng.gamma(2.0, (mean[:, None] + 0.5) / 2.0, (len(mean), k))  # per-sample KL: noisy, positive


def cells_from_latent(rng, lat, k=4, nU=4, seeds=SEEDS):
    cells = {"U": _bin_cell(rng, "U", None, [lat["U"]] * nU, k, range(nU))}
    cells["U"].lineage = None
    N = len(lat["U"])
    ids = list(range(N))
    for src in SRCS:
        pre = _bin_cell(rng, f"{src}|pre", src, [lat[(src, None, "pre", s)] for s in seeds], k, seeds)
        for tag in TAGS:
            cells[f"{src}|pre@{tag}"] = pre
    for arm in ARMS:
        for tag in TAGS:
            for t in STEPS:
                cells[f"{arm}@{tag}|{t}"] = _bin_cell(rng, f"{arm}@{tag}|{t}", arm, [lat[(arm, tag, t, s)] for s in seeds], k, seeds)
                cells[f"KL:{arm}@{tag}|{t}"] = kl_cell(f"KL:{arm}@{tag}|{t}", arm,
                                                       [(s, (ids, k, _kl_draw(rng, lat[("KL", arm, tag, t, s)], k))) for s in seeds], "gsm8k")
    return cells


def _dry_job(job):
    world, seed, B, nsim = job
    rng = np.random.default_rng(seed)
    d = L.Design(cells_from_latent(rng, kl_latent(rng, world)))
    obs, reps = L.bootstrap(d, stats19_fn(graders=("len",), steps=STEPS), B, seed=seed + 1)
    ci = lambda k: L.ci(reps[k]) if k in reps else (NAN, NAN)
    r = {}
    for src in SRCS:
        g = lambda k: obs.get(f"len|{src}|{k}", NAN)
        r[f"{src} b300 beta=0"], r[f"{src} b300 beta=0.5"] = obs.get(f"len|300|{src}@0|b", NAN), obs.get(f"len|300|{src}@0.5|b", NAN)
        r[f"{src} rho_beta"] = g("rho_beta_b300")
        r[f"{src} rule rho_beta >= 0.8"] = float(g("rho_beta_b300") >= REG19["RHO_BETA"])
        r[f"{src} rule rho_beta = 1"] = float(g("rho_beta_b300") >= 0.999)
        lo, _ = ci(f"len|{src}|b300_hi_minus_lo")
        r[f"{src} b(0.5)-b(0)"] = g("b300_hi_minus_lo")
        r[f"{src} rule diff >= 0.15 & CI > 0"] = float(g("b300_hi_minus_lo") >= REG19["DIFF"] and lo > 0)
        r[f"{src} rule diff CI > 0"] = float(lo > 0)
        for lab in ("S", "R"):
            r[f"{src} rho_erasure_kl{lab} (12 cells)"] = g(f"rho_erasure_kl{lab}")
            r[f"{src} rho_erasure_kl{lab} by step"] = g(f"rho_erasure_kl{lab}_bystep")
        r[f"{src} rule rho_klS 12 cells >= 0.6"] = float(g("rho_erasure_klS") >= REG19["RHO_KL"])
        r[f"{src} rule rho_klS by step >= 0.6"] = float(g("rho_erasure_klS_bystep") >= REG19["RHO_KL"])
        lo2, _ = ci(f"len|{src}|rho_erasure_klS_bystep")
        r[f"{src} rule rho_klS by step CI > 0"] = float(lo2 > 0)
        r[f"{src} erasure excess at matched KL"] = g("erasure_excess_matchedKL")
        r[f"{src} matched-KL cells"] = g("n_matchedKL_cells")
        xlo, xhi = ci(f"len|{src}|erasure_excess_matchedKL")
        r[f"{src} rule matched-KL excess CI < 0"] = float(xhi < 0)
        q = quad19(d.cells, src, "0.5", 300)
        nl, _ = tilt_null(d, "len", q, nsim, seed + 2 + (src == "teacher"))
        nb = nl["b"][np.isfinite(nl["b"])]
        lo_, hi_ = np.percentile(nb, [2.5, 97.5])
        b5 = obs.get(f"len|300|{src}@0.5|b", NAN)
        r[f"{src} null median b(0.5)"] = med(nb)
        r[f"{src} rule b(0.5) inside null band"] = float(lo_ <= b5 <= hi_)
    return world, r


def dry_synthetic(args):
    P(f"# DRY RUN 19 M5 (synthetic worlds; GSM8K sizes: 1,319 items, k = 4 pre/post, untrained 16 samples, 3 seeds; "
      f"{args.power_reps} datasets per world; bootstrap B = {args.dry_B}; tilt null {args.dry_nsim} replicates)")
    for w, txt in WORLDS.items():
        P(f"  {w}: {txt}")
    jobs = [(w, 19000 + 1000 * i + r, args.dry_B, args.dry_nsim) for i, w in enumerate(WORLDS) for r in range(args.power_reps)]
    with Pool(args.procs) as pool:
        out = pool.map(_dry_job, jobs, chunksize=1)
    for w in WORLDS:
        rr = [x for ww, x in out if ww == w]
        P(f"\n## world '{w}' ({len(rr)} datasets)")
        for k in rr[0]:
            v = np.array([x[k] for x in rr], float)
            P(f"    {k:42s} mean {np.nanmean(v):+.3f} SD {np.nanstd(v):.3f}" + (f"   P(rule) {np.nanmean(v > 0.5):.2f}" if " rule " in k else ""))


# ---------------------------------------------------------------- mock files (end-to-end test of the loaders and reports)
def _rows_probe(rng, adapter, p, gold, k, extra=None, math=False):
    out = []
    for i, (pi, g) in enumerate(zip(p, gold)):
        for j in range(k):
            ok = rng.random() < pi
            ans = g if ok else str(int(g) + 7)
            comp = (f"so the answer is \\boxed{{{ans}}}" if math else f"so #### {ans}")
            out.append({"adapter": adapter, "example_id": i, "sample": j, "gold": g, "correct": bool(ok), "stopped": True,
                        "completion": comp, **(extra or {})})
    return out


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def _rollouts(path, trunc=0.0, rng=None, bad_tail=False, steps=300):
    rows = []
    for st in range(steps):
        for _ in range(4):
            stopped = not (rng.random() < trunc) and not (bad_tail and st >= steps - 30)
            rows.append({"step": st, "reward": 0.5, "stopped": stopped})
    _write(path, rows)


def write_mock(root, rng, N=400, world="switch"):
    root = Path(root)
    lat = kl_latent(rng, world, N=N)
    gold = [str(1000 + i) for i in range(N)]
    _write(root / "runs/probes/prereg14_a/untrained.jsonl", _rows_probe(rng, None, lat["U"], gold, 4))
    for s in SEEDS:
        rows = [{k_: v for k_, v in r.items() if k_ != "adapter" and k_ != "example_id"} for r in _rows_probe(rng, None, lat["U"], gold, 4)]
        _write(root / f"runs/gsm8k/Llama-3.2-1B-Instruct-censored-grpo/seed_{s}/base_predictions.jsonl", rows)
    G = "runs/gsm8k/Llama-3.2-1B-Instruct"
    start = []
    for src in SRCS:
        for s in SEEDS:
            start += _rows_probe(rng, f"{G}{'-qwen7b' if src == 'teacher' else ''}/seed_{s}/random_sft_250/sft", lat[(src, None, "pre", s)], gold, 4, {"merged_start": True})
    _write(root / PDIR / "start.jsonl", start)
    for tag in TAGS:
        for t in STEPS:
            rows, klr = [], []
            for arm in ARMS:
                for s in SEEDS:
                    a = str(run_dir("", arm, tag, s) / f"grpo/checkpoint-{t}").lstrip("/")
                    rows += _rows_probe(rng, a, lat[(arm, tag, t, s)], gold, 4)
                    kd = _kl_draw(rng, lat[("KL", arm, tag, t, s)], 4)
                    klr += [{"adapter": a, "example_id": i, "sample": j, "kl@0.6": float(kd[i, j]), "lr@0.6": float(kd[i, j] + rng.normal()),
                             "kl@1": float(1.3 * kd[i, j]), "lr@1": 0.0} for i in range(N) for j in range(4)]
            _write(root / PDIR / f"b{tag}_s{t}.jsonl", rows)
            _write(root / PDIR / f"kl_b{tag}_s{t}.jsonl", klr)
        for arm in ARMS:
            for s in SEEDS:
                d = run_dir(root, arm, tag, s)
                _rollouts(d / "grpo/rollouts.jsonl", 0.02, rng, bad_tail=(arm == "teacher" and tag == "0.5" and s == 44))
                strip = lambda rr: [{k_: v for k_, v in r.items() if k_ not in ("adapter", "example_id")} for r in rr]
                _write(d / "predictions.jsonl", strip(_rows_probe(rng, None, lat[(arm, tag, 300, s)], gold, 4)))
                if arm != "grpo":
                    _write(d / "predictions_after_sft.jsonl", strip(_rows_probe(rng, None, lat[(arm, None, "pre", s)], gold, 4)))
    # M6 (17B Gemma + dr_grpo), smaller: 200 items, k = 8 (untrained 16)
    M = 200
    goldm = [str(2000 + i) for i in range(M)]
    pU = L.expit(rng.normal(-1.2, 2.0, M))
    pre = {(a, s): L.expit(np.log(pU / (1 - pU)) - 1.0 + rng.normal(0, 0.3, M)) for a in DR_ARMS for s in DR_SEEDS}
    _write(root / "runs/probes/prereg17b/gemma_mf_untrained.jsonl", _rows_probe(rng, None, pU, goldm, 16, math=True))
    _write(root / "runs/probes/prereg17b/gemma_mf_sft.jsonl", sum((_rows_probe(rng, f"runs/math35d/gemma-p17b-mf-{a}/seed_{s}/listed_sft/sft", pre[(a, s)], goldm, 8, math=True)
                                                                    for a in DR_ARMS for s in DR_SEEDS), []))
    for t in DR_STEPS:
        r17, rdr = [], []
        for s in SEEDS:
            r17 += _rows_probe(rng, f"runs/math35d/gemma-p17b-mf-grpo/seed_{s}/full_grpo/grpo/checkpoint-{t}", L.expit(np.log(pU / (1 - pU)) + 0.5), goldm, 8, math=True)
        for a in DR_ARMS:
            for s in DR_SEEDS:
                r17 += _rows_probe(rng, f"runs/math35d/gemma-p17b-mf-{a}/seed_{s}/listed_sft/grpo/checkpoint-{t}", L.expit(np.log(pre[(a, s)] / (1 - pre[(a, s)])) + 0.2), goldm, 8, math=True)
                rdr += _rows_probe(rng, f"runs/math35d/gemma-p19dr-mf-{a}/seed_{s}/listed_sft/grpo/checkpoint-{t}", L.expit(np.log(pre[(a, s)] / (1 - pre[(a, s)])) + 0.6), goldm, 8, math=True)
        _write(root / f"runs/probes/prereg17b/gemma_mf_s{t}.jsonl", r17)
        _write(root / P19DR / f"gemma_dr_s{t}.jsonl", rdr)
    for a in DR_ARMS + ("grpo",):
        for s in SEEDS:
            _rollouts(dr_dir(root, a, s, False) / "grpo/rollouts.jsonl", 0.05 if a == "grpo" else 0.6, rng)
            if a != "grpo" and s in DR_SEEDS:
                _rollouts(dr_dir(root, a, s, True) / "grpo/rollouts.jsonl", 0.08, rng)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["M5", "M6"])
    ap.add_argument("--dry_run", choices=["synthetic", "mock"])
    ap.add_argument("--schema_only", action="store_true")
    ap.add_argument("--unblind", action="store_true", help="real scoring; refused unless prereg/19_*.md exists")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--kl_key", default="kl@0.6", help="kl_travel.py column: kl@0.6 (primary), kl@1, lr@0.6, lr@1")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--nsim", type=int, default=200)
    ap.add_argument("--power_reps", type=int, default=60)
    ap.add_argument("--dry_B", type=int, default=200)
    ap.add_argument("--dry_nsim", type=int, default=100)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    P(f"# score_prereg19_kl.py args={vars(args)}\n# REG19 (candidates)={REG19}\n# frozen libraries: {L.__file__}, "
      f"{S.__file__}, {S17.__file__}")
    t0 = time.time()
    if args.dry_run == "synthetic":
        dry_synthetic(args)
    elif args.dry_run == "mock":
        mroot = HERE / "p19_mock"
        write_mock(mroot, np.random.default_rng(1919))
        P(f"# mock files written under {mroot} (synthetic 'switch' world; teacher b0.5 seed 44 has a truncated tail -> excluded)")
        schema19(mroot, "M5")
        part_m5(mroot, args)
        schema19(mroot, "M6")
        part_m6(mroot, args)
    elif args.schema_only:
        schema19(args.root, args.part)
    else:
        if not args.unblind or not list((Path(args.root) / "prereg").glob("19_*.md")):
            sys.exit("refused: pass --unblind, and only after prereg/19_*.md is registered (committed) and the runs are finished")
        (part_m5 if args.part == "M5" else part_m6)(args.root, args)
    P(f"\n# done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
