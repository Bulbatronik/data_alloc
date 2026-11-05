#!/usr/bin/env python3
"""score_prereg19_grad.py -- scorer for prereg 19's gradient experiments M3 and M4 (runs/reviews/round4/MECH_PLAN.md).
Written and dry-run BEFORE prereg 19 and before any M3/M4 statistic on real data. CPU only (Slurm), numpy (+ safetensors).

Modes
  --schema_only            file counts / keys / shapes / item alignment; never touches correctness or gradients' values
  --dry_run synthetic      planted-effect simulations: SNR estimator, M3 rules (FPR, power), M4 estimators (sketch
                           debiasing, noise), M4 (b) false-positive rates and power, bootstrap coverage
  --dry_run mock --root R  end-to-end on a mock root (R built by the frozen prereg/scripts/p15/p15_mock.py) with
                           synthetic galign / M3 files written here: loaders, id alignment, frozen Part 1 cross-check
  --unblind                real scoring; refused unless a prereg/19_*.md has a '## Part IIb' section naming this scorer
  --export_p1 BASE         format conversion only: per-direction frozen Part 1 input dirs BASE_<direction>

M3 (firstgrad.py --n_batches --adam_dirs; exact phi by m3phi.py, phix_<dtag>_A.jsonl): direction families raw_g<G>
  (v0_lora_g<G>), sign_g<G>, adamv_g<G> (gbar / (sqrt(vhat_G) + eps)), adamv25_g<G> (the run's own step-25 second
  moment), adam_run1/2 (the run's own first Adam steps), each with matched random controls craw/csign/cadamv/cadamv25
  (random LoRA-geometry directions through the same transform; craw_1 = Part 1's rand_lora). For each direction the
  registered Part 1 statistics (frozen score_prereg15.run_part1, primary gold_mf, set A/B/C of Part 1 via
  lead_p1_convert's part1_reg_i42) with v0 := the direction, rand := its controls, placebo = the untrained start's same
  direction. Joint statistics (one bootstrap: items, then gold starts, as run_part1 resamples): K1 / K2 per start and
  pooled, margins over controls, placebo and raw (same G), trends over G per family; the Adam rule (K1' and >= D_RAW
  below raw). Gradient SNR from the per-batch Gram matrices
  (U-statistics: |mu|^2 = mean off-diagonal Gram, tr Sigma = mean diagonal - |mu|^2; batch bootstrap), in the merged
  v0_lora geometry (registered), the trainable-B geometry and the dense raw geometry (point estimates); cosines of the
  G-batch mean with v25 / v150 / Part 1's v0 / the SFT displacement, raw and disattenuated (cos(mu_hat, ref)).
M4 (gradient alignment; galign.py): g(x) = grad_B logit p_T(x) = mean_correct g_i - mean_incorrect g_i (score-function
  identity; T 0.6). P = projector onto the top-r right-singular subspace of the sketched pool matrix (rows: the start's
  RL-pool prompts' g, weighted sqrt(p(1-p)) as GRPO weights them). Sketch inflation k/d is removed:
  |P S u|^2 -> (|P S u|^2 - (k/d)|u|^2) / (1 - k/d) with |u|^2 exact from the Gram. Per item:
    a_naive(x)^2  from all K samples (sketch-debiased; still attenuated by sampling noise in g)
    num(x)        <P g^(1), P g^(2)> from disjoint sample halves (unbiased for |P g(x)|^2), den(x) = <g^(1), g^(2)>
  (a) A_set = sqrt(sum num / sum den) (cross-fitted, noise-free in expectation); ratio A_off / A_in, OOD pooled at item
      level (GSM-Plus + SVAMP + MATH-600) vs GSM8K test; per-set ratios; the naive ratio is reported.
  (b) registered candidate (REG19 B_STAT): the tercile contrast of Delta D^(1) Delta D^(2) (unbiased for Delta D^2)
      between the top and bottom thirds of a_naive WITHIN strata of the start's galign count n1 (which sets a_naive's
      noise attenuation), pooled over the four sets (item-weighted). Reported: per set; unstratified; partial Spearman
      given count dummies; plain and linear-partial Spearman(a_naive, |Delta D|) (the plan's literal statistic: NOT valid,
      false-positive rate 0.08-0.28 in dry run S4).
      Delta D = (q_S - q_R) - (p_S - p_U) per item from the frozen prereg-15/16 loaders (part_16's paired composition;
      GSM8K: Part A cells with E2's untrained samples pooled). CIs: hierarchical (seeds within lineage, then items;
      GSM-Plus clustered with its GSM8K seed question; L.bootstrap) and item-only.
"""
import argparse
import contextlib
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "prereg/scripts/p15"))
import p15lib as L  # noqa: E402
import score_prereg15 as S15  # noqa: E402

FROZEN_SHA = {
    "prereg/scripts/p15/score_prereg15.py": "9def2afa975ecfca4ad8769f23fdc6cf7c82ca275da4da5ab21be6b48ee32e31",
    "prereg/scripts/p15/p15lib.py": "9c15684cbdd0ec13e00f6cf36034b475ad51fb22a2b16dae97413735606edf0d",
    "prereg/scripts/p16/score_prereg15.py": "668a14a57c68a83e0b605f2ef2733a77ba2b22267b0c4e720d6e3acfc3c04dab",
    "prereg/scripts/p16/p15lib.py": "9c15684cbdd0ec13e00f6cf36034b475ad51fb22a2b16dae97413735606edf0d",
}
LOADERS_16 = ("load_A", "part_A_exclusions", "cells_A", "load_16", "gsm_plus_clusters", "build_16_design", "hf_rows",
              "gsm8k_gold", "parse_gsm8k_adapter", "stop_rate_trainer_state")

# ---------------------------------------------------------------- proposed registered constants (lead fixes them)
REG19 = dict(
    # M3
    G_LIST=(1, 4, 16, 64, 256), G_PASS=(16, 64), G_PRIMARY=256, ADAM_PRIMARY="adamv",
    K1_MAX=S15.REG["K1_MAX"], K_MARGIN=S15.REG["K_MARGIN"], K2_MIN=S15.REG["K2_MIN"], MIN_PASS=2,
    D_RAW=0.08,                                          # Adam beats raw by >= D_RAW in K1 (dry run S2b: FPR <= 0.05)
    SNR_WELL=4.0,                                        # SNR_G >= 4 <=> E cos(mean_G, mu) >= 0.89
    C_PHI=0.37,                                          # Part 1's measured phi/generation cost (crossover print only)
    # M4
    D=8192, R=64, D_LIST=(2048, 4096, 8192), R_LIST=(16, 64, 256, 0),  # r = 0: the full span of the pool rows
    RATIO_MIN=0.8, POOL_WEIGHT="grpo", TEMP=0.6, ARM="gold_250", GRADER="len", B_STAT="b_tercile_strata",
)
SEEDS = (42, 43, 44)
M3_STARTS = {"gold_s42_merged_i42": "gold_mf_s42", "gold_s43_merged_i43": "gold_mf_s43",
             "gold_s44_merged_i44": "gold_mf_s44", "untrained_i42": "untrained"}
M4_STARTS = {"gold_s42_merged_i42": "gold", "untrained_i42": "untrained"}
GA_SETS = ("gsm8k", "gsm_plus", "svamp", "math600")
OOD = ("gsm_plus", "svamp", "math600")
GOLD_PARTITION = "runs/gsm8k/Llama-3.2-1B-Instruct/seed_42/random_sft_250/partition.json"


def P(*a):
    print(*a, flush=True)


fmt = S15.fmt


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def func_src(text, name):
    m = re.search(rf"^def {name}\(.*?(?=^def |^class |^# =+|\Z)", text, re.S | re.M)
    return m.group(0).rstrip() if m else None


def check_frozen():
    """The frozen scorers are unchanged, and prereg 16's loaders (p16 copy) equal the p15 copy imported here."""
    for rel, h in FROZEN_SHA.items():
        got = sha256(ROOT / rel)
        assert got == h, f"{rel}: sha256 {got[:8]} != registered {h[:8]}"
    t15 = (ROOT / "prereg/scripts/p15/score_prereg15.py").read_text()
    t16 = (ROOT / "prereg/scripts/p16/score_prereg15.py").read_text()
    for fn in LOADERS_16:
        a, b = func_src(t15, fn), func_src(t16, fn)
        assert a is not None and a == b, f"loader {fn} differs between the frozen p15 and p16 scorers"
    P(f"# frozen scorers verified (sha256) and prereg-16 loaders identical in p15/p16: {', '.join(LOADERS_16)}")


def blind_gate():
    """M3/M4 are prereg 19 Part IIb. prereg/19_mechanisms.md already registers Parts I and IIa, so a file's existence is
    not enough: --unblind needs a '## Part IIb' section in a prereg/19_*.md that names this scorer."""
    for r in sorted((ROOT / "prereg").glob("19_*.md")):
        text = r.read_text()
        if re.search(r"^##\s+Part IIb", text, re.M) and "score_prereg19_grad" in text:
            P(f"# registration found: {r.relative_to(ROOT)} (Part IIb, names score_prereg19_grad)")
            return
    sys.exit("blinded: no '## Part IIb' section naming score_prereg19_grad in prereg/19_*.md; --unblind is refused")


def st_load(path):
    from safetensors.numpy import load_file
    return load_file(str(path))


def wmean(vals, w):
    v, w = np.asarray(vals, float), np.asarray(w, float)
    ok = np.isfinite(v) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def spear(x, y):
    return L.pspear(y, x, [])


# =========================================================================================================== M3
# Directions (phix_<dtag>_A.jsonl from m3phi.py, exact phi; names "<base>_<dtag>") -> scorer names
PHIX_MAP = [(r"v0_lora_g(\d+)", "raw_g{}"), (r"sign_g(\d+)", "sign_g{}"), (r"adamv_g(\d+)", "adamv_g{}"),
            (r"adamv25_g(\d+)", "adamv25_g{}"), (r"adam_run([12])", "adam_run{}"), (r"p1_v0_lora", "v0_p1x"),
            (r"p1_rand_lora", "craw_1"), (r"rand_lora_r(\d+)", "craw_{}"), (r"rand_sign_r(\d+)", "csign_{}"),
            (r"rand_adamv_r(\d+)", "cadamv_{}"), (r"rand_adamv25_r(\d+)", "cadamv25_{}")]
FAMILIES = ("raw", "sign", "adamv", "adamv25")
CONTROL = {"raw": "craw", "sign": "csign", "adamv": "cadamv", "adamv25": "cadamv25", "adam_run": "csign"}


def fam_of(dn):
    m = re.fullmatch(r"(raw|sign|adamv|adamv25)_g\d+", dn)
    if m:
        return m.group(1)
    return "adam_run" if dn.startswith("adam_run") else None


def g_of(dn):
    m = re.search(r"_g(\d+)$", dn)
    return int(m.group(1)) if m else None


def map_phix(base):
    for rx, tmpl in PHIX_MAP:
        m = re.fullmatch(rx, base)
        if m:
            return tmpl.format(*m.groups())
    return None


def m3_load(p1_reg, m3_dir, dtags=M3_STARTS):
    """rows (A, B, C) for p1_arrays. Part 1 rows (FD phi): v0 -> v0_p1, rand_1 -> rand_p1, v25, v150 (v0raw dropped).
    M3 rows (exact phi, m3phi.py): raw_g<G>, sign_g<G>, adamv_g<G>, adamv25_g<G>, adam_run1/2, v0_p1x (Part 1's v0, exact)
    and the matched random controls craw_<j> (craw_1 = Part 1's rand_lora), csign_<j>, cadamv_<j>, cadamv25_<j>."""
    p1_reg, m3_dir = Path(p1_reg), Path(m3_dir)
    rows = {s: list(L.read_jsonl(p1_reg / f"{s}.jsonl")) for s in "BC"}
    rows["A"] = []
    ren = {"v0": "v0_p1", "rand_1": "rand_p1"}
    for r in L.read_jsonl(p1_reg / "A.jsonl"):
        dn = r["direction_name"]
        if dn != "v0raw":
            rows["A"].append({**r, "direction_name": ren.get(dn, dn)})
    found, grams, adam = {}, {}, {}
    for dtag, st in dtags.items():
        f = m3_dir / f"phix_{dtag}_A.jsonl"
        if not f.exists():
            P(f"  [m3] missing {f}")
            continue
        for r in L.read_jsonl(f):
            assert r["direction_name"].endswith("_" + dtag), r["direction_name"]
            dn = map_phix(r["direction_name"][: -len(dtag) - 1])
            if dn is None:
                continue
            found.setdefault(st, set()).add(dn)
            rows["A"].append({"item": r["item"], "sample": r["sample"], "start": st, "direction_name": dn,
                              "phi": float(r["phi"]), "correct_lenient": r["correct_lenient"],
                              "correct_marked": r["correct_marked"], "set": "A"})
        g = m3_dir / "directions" / dtag / "m3_gram.safetensors"
        if g.exists():
            grams[st] = (st_load(g), json.loads(g.with_suffix(".json").read_text()))
        a = m3_dir / "directions" / dtag / "m3_adam.json"
        if a.exists():
            adam[st] = json.loads(a.read_text())
    covs = list(L.read_jsonl(p1_reg / "covariates.jsonl")) if (p1_reg / "covariates.jsonl").exists() else []
    return rows, covs, grams, adam, {k: sorted(v) for k, v in found.items()}


def rows_for(rows, dn_v0, ctrl):
    """The frozen Part 1 input for one direction: v0 := dn_v0; rand_<j> := the matched controls <ctrl>_<j>, rescaled to
    v0's RMS phi per start (Spearman-invariant; keeps run_part1's 'matched' print true); v25, v150 kept."""
    sq = defaultdict(list)
    for r in rows["A"]:
        sq[(r["start"], r["direction_name"])].append(r["phi"] ** 2)
    rms = {k: math.sqrt(np.mean(v)) for k, v in sq.items()}
    A = []
    for r in rows["A"]:
        dn, st = r["direction_name"], r["start"]
        if dn == dn_v0:
            A.append({**r, "direction_name": "v0"})
        elif re.fullmatch(rf"{ctrl}_\d+", dn) and (st, dn_v0) in rms:
            A.append({**r, "direction_name": "rand_" + dn.split("_")[-1],
                      "phi": r["phi"] * rms[(st, dn_v0)] / max(rms[(st, dn)], 1e-300)})
        elif dn in ("v25", "v150"):
            A.append(r)
    return {"A": A, "B": rows["B"], "C": rows["C"]}


@contextlib.contextmanager
def capture_frozen():
    buf, orig = [], S15.P

    def P2(*a):
        orig(*a)
        buf.append(" ".join(map(str, a)))
    S15.P = P2
    try:
        yield buf
    finally:
        S15.P = orig


def parse_frozen(lines, grader="len"):
    """K1|.. / K2|.. values printed by the frozen run_part1 in its [grader] section (3 decimals)."""
    out, cur = {}, None
    for ln in lines:
        h = re.search(r"## \[(len|mk)\]", ln)
        if h:
            cur = h.group(1)
            continue
        m = re.match(r"^\s+((?:K1|K2)\|\S+)\s+([+-]?\d+\.\d+|nan)", ln)
        if m and cur == grader:
            out[m.group(1)] = float(m.group(2))
    return out


def m3_stats_fn(arrs, golds, dirs, ctrls, refs=("v25", "v150", "v0_p1", "rand_p1", "v0_p1x")):
    """dirs: the tested directions (family_g<G>, adam_run1/2); ctrls: {control prefix: [names]}. Statistics replicate
    the frozen run_part1 `stats` (K1 = partial Spearman(Delta^v, D_SFT | p_S); K2 with p_S, p_U, D_SFT, f, covariates)
    per start, then pool over gold starts with the bootstrap start weights w."""
    items, succ, delta, sbar2, phi, f_unt, cv = arrs
    pA = lambda st: succ[("A", st)][0]
    pB = lambda st: succ[("B", st)][0]
    allc = [c for cs in ctrls.values() for c in cs]

    def fn(J, w):
        out = {}
        for st in golds:
            dsft = (pB(st) - pB("untrained"))[J]
            pS, p0 = pA(st)[J], pA("untrained")[J]
            for dn in list(dirs) + allc + list(refs):
                if (st, dn) in delta:
                    out[f"K1|{st}|{dn}"] = L.pspear(delta[(st, dn)][J], dsft, [pS])
            for dn in dirs:
                if ("untrained", dn) in delta:
                    out[f"K1|{st}|placebo_{dn}"] = L.pspear(delta[("untrained", dn)][J], dsft, [pS])
            for c, cs in ctrls.items():
                out[f"K1|{st}|ctrl_{c}"] = np.nanmean([out.get(f"K1|{st}|{x}", np.nan) for x in cs]) if cs else np.nan
            post = succ.get(("C", f"post_{st}"))
            if post is not None:
                y = (post[0] - pB(st))[J]
                Z = [pS, p0, dsft, f_unt[J]] + [cv[(st, nm)][J] for nm in ("silhouette", "length", "entropy") if (st, nm) in cv]
                for dn in list(dirs) + allc:
                    if (st, dn) in delta:
                        out[f"K2|{st}|{dn}"] = L.pspear(delta[(st, dn)][J], y, Z)
                for c, cs in ctrls.items():
                    out[f"K2|{st}|ctrl_{c}"] = np.nanmean([out.get(f"K2|{st}|{x}", np.nan) for x in cs]) if cs else np.nan
        for dn in dirs:
            fam = fam_of(dn)
            c = CONTROL[fam]
            k1 = [out.get(f"K1|{st}|{dn}", np.nan) for st in golds]
            out[f"K1bar|{dn}"] = wmean(k1, w)
            out[f"margin_ctrl|{dn}"] = wmean([a - out.get(f"K1|{st}|ctrl_{c}", np.nan) for a, st in zip(k1, golds)], w)
            out[f"margin_placebo|{dn}"] = wmean([a - out.get(f"K1|{st}|placebo_{dn}", np.nan) for a, st in zip(k1, golds)], w)
            out[f"K2bar|{dn}"] = wmean([out.get(f"K2|{st}|{dn}", np.nan) for st in golds], w)
            out[f"K2margin_ctrl|{dn}"] = wmean([out.get(f"K2|{st}|{dn}", np.nan) - out.get(f"K2|{st}|ctrl_{c}", np.nan)
                                               for st in golds], w)
            G = g_of(dn)
            if fam not in ("raw", "adam_run") and f"raw_g{G}" in dirs:
                out[f"margin_raw|{dn}"] = wmean([a - out.get(f"K1|{st}|raw_g{G}", np.nan) for a, st in zip(k1, golds)], w)
        for fam in FAMILIES:
            Gs = sorted(g_of(d) for d in dirs if fam_of(d) == fam)
            k = np.array([out.get(f"K1bar|{fam}_g{G}", np.nan) for G in Gs])
            if len(Gs) >= 2 and np.all(np.isfinite(k)):
                x = np.log2(np.array(Gs, float))
                out[f"trend_slope_log2G|{fam}"] = float(np.polyfit(x, k, 1)[0])
                out[f"trend_spearman_G|{fam}"] = float(np.corrcoef(L.rank(x), L.rank(k))[0, 1]) if np.ptp(k) > 0 else np.nan
                out[f"K1bar_Gmax_minus_G1|{fam}"] = float(k[-1] - k[0])
        return out
    return fn


def k1_pass(obs, st, dn):
    """The frozen K1' rule for one start and direction, against the direction's matched controls and its placebo."""
    s1 = obs.get(f"K1|{st}|{dn}", np.nan)
    c = CONTROL[fam_of(dn)]
    return bool(s1 <= REG19["K1_MAX"] and s1 <= obs.get(f"K1|{st}|ctrl_{c}", np.nan) - REG19["K_MARGIN"]
                and s1 <= obs.get(f"K1|{st}|placebo_{dn}", np.nan) - REG19["K_MARGIN"])


def adam_pass(obs, st, dn):
    """Prereg-19 Adam rule: the K1' rule for an optimiser-geometry direction AND at least D_RAW below the raw direction
    of the same G at the same start."""
    G = g_of(dn)
    raw = obs.get(f"K1|{st}|raw_g{G}", np.nan)
    return bool(k1_pass(obs, st, dn) and obs.get(f"K1|{st}|{dn}", np.nan) <= raw - REG19["D_RAW"])


def gram_snr(K, idx):
    """U-statistic SNR from a batch Gram matrix restricted to (possibly repeated) batch indices idx: pairs of the same
    original batch are excluded from |mu|^2."""
    Ks = K[np.ix_(idx, idx)]
    off = idx[:, None] != idx[None, :]
    if off.sum() == 0:
        return np.nan, np.nan, np.nan
    mu2 = Ks[off].mean()
    tr = np.diag(Ks).mean() - mu2
    return float(mu2), float(tr), float(mu2 / tr) if tr > 0 else np.nan


def m3_gradient_report(grams, Gs, B=2000, seed=1911):
    """SNR of the raw step-0 gradient, the G-batch cosines with the references, disattenuated cosines (batch bootstrap)."""
    res = {}
    rng = np.random.default_rng(seed)
    for st, (t, meta) in sorted(grams.items()):
        KW, KB, ip = t["gram_W"], t["gram_B"], t["ip_W"]
        nb = KW.shape[0]
        refs, rn = meta["refs"], meta["ref_norms"]
        allb = np.arange(nb)
        o = {"n_batches": nb, "refs": refs}
        for geo, K in (("W", KW), ("B", KB)):
            mu2, tr, s1 = gram_snr(K, allb)
            reps = np.array([gram_snr(K, rng.integers(0, nb, nb))[2] for _ in range(B)])
            o[f"snr1_{geo}"] = (s1, L.ci(reps))
            o[f"mu2_{geo}"], o[f"trSigma_{geo}"] = mu2, tr
            for G in Gs:
                o[f"snrG{G}_{geo}"] = (G * s1, L.ci(G * reps))
                o[f"Ecos_mu_G{G}_{geo}"] = math.sqrt(G * s1 / (1 + G * s1)) if s1 > 0 else np.nan
        if meta.get("raw_stats") and "raw_sq" in t:
            sq, cr = t["raw_sq"], t["raw_cross"]
            mu2 = 2 * cr.sum() / (nb * (nb - 1))
            tr = sq.mean() - mu2
            o["snr1_raw"] = (mu2 / tr if tr > 0 else np.nan, (np.nan, np.nan))
        for k, r in enumerate(refs):
            nr = rn[r]
            for G in sorted(set(Gs) | {nb}):
                if G <= nb:
                    idx = np.arange(G)
                    o[f"cos|{r}|G{G}"] = ip[idx, k].mean() / (math.sqrt(max(KW[np.ix_(idx, idx)].mean(), 1e-300)) * nr)
            mu2 = gram_snr(KW, allb)[0]

            def dis(idx):
                m2 = gram_snr(KW, idx)[0]
                return ip[idx, k].mean() / (math.sqrt(m2) * nr) if m2 > 0 else np.nan
            reps = np.array([dis(rng.integers(0, nb, nb)) for _ in range(B)])
            o[f"cos_mu|{r}"] = (ip[:, k].mean() / (math.sqrt(mu2) * nr) if mu2 > 0 else np.nan, L.ci(reps))
            cs = [o[f"cos|{r}|G{G}"] for G in Gs if G <= nb]
            o[f"cos_rises|{r}"] = bool(len(cs) >= 2 and all(b > a for a, b in zip(cs, cs[1:])))
        half = nb // 2
        if half >= 1:
            a, b = np.arange(half), np.arange(half, 2 * half)
            o["split_half_cos_W"] = KW[np.ix_(a, b)].mean() / math.sqrt(KW[np.ix_(a, a)].mean() * KW[np.ix_(b, b)].mean())
        o["mean_reward"] = float(np.mean([x["mean_reward"] for x in meta["batches"]]))
        o["frac_zero_adv_groups"] = float(np.mean([x["frac_zero_adv_groups"] for x in meta["batches"]]))
        res[st] = o
    return res


def print_gradient_report(res, Gs):
    for st, o in res.items():
        P(f"  [{st}] batches {o['n_batches']}; mean reward {o['mean_reward']:.3f}; zero-advantage groups "
          f"{o['frac_zero_adv_groups']:.3f}; split-half cos (W) {o.get('split_half_cos_W', np.nan):+.3f}")
        for geo in ("W", "B"):
            s1, ci = o[f"snr1_{geo}"]
            P(f"    SNR per batch [{geo}] {s1:.4g} [{ci[0]:.4g}, {ci[1]:.4g}]  |mu|^2 {o[f'mu2_{geo}']:.4g}  trSigma "
              f"{o[f'trSigma_{geo}']:.4g}; " + "; ".join(f"G{G}: SNR {o[f'snrG{G}_{geo}'][0]:.3g}, E cos(mean, mu) "
                                                           f"{o[f'Ecos_mu_G{G}_{geo}']:.3f}" for G in Gs))
        if "snr1_raw" in o:
            P(f"    SNR per batch [raw dense, point] {o['snr1_raw'][0]:.4g}")
        for r in o["refs"]:
            cs = "  ".join(f"G{G} {o[f'cos|{r}|G{G}']:+.4f}" for G in Gs if f"cos|{r}|G{G}" in o)
            c, ci = o[f"cos_mu|{r}"]
            P(f"    cos(raw mean_G, {r}): {cs};  rises with G: {o[f'cos_rises|{r}']};  disattenuated cos(mu, {r}) "
              f"{fmt(c, ci, 4)}")


def print_adam_cosines(adam, grams):
    """Weight-space cosines of every saved direction with the references (point estimates, from m3_adam.json)."""
    for st, a in sorted(adam.items()):
        rn = grams[st][1]["ref_norms"] if st in grams else {}
        refs = [r for r in a["refs"] if r in rn]
        P(f"  [{st}] cosines with {refs} (clip coefficients < 1 in {sum(c < 1 for c in a['clip_coef'])} of "
          f"{len(a['clip_coef'])} batches; run-batch reconstruction {a.get('run_step0_grad_norm', 'n/a')})")
        for name in sorted(a["directions"], key=lambda x: (x.split("_g")[0], g_of(x) or 0)):
            d = a["directions"][name]
            cs = "  ".join(f"{r} {d['ip'][r] / (d['norm'] * rn[r]):+.4f}" for r in refs)
            P(f"    {name:20s} {cs}")


def score_m3(p1_reg, m3_dir, B, seed=1910, check_frozen_values=True, c_phi=REG19["C_PHI"], frozen_fams=("raw", "adamv")):
    P(f"\n# M3 — expected step-0 gradient and optimiser geometry; Part 1 sets from {p1_reg}; M3 from {m3_dir}")
    rows, covs, grams, adam, found = m3_load(p1_reg, m3_dir)
    if not found:
        P("  no M3 phi files: M3 not scored")
        return {}
    common = set.intersection(*(set(v) for v in found.values()))
    dirs = sorted((d for d in common if fam_of(d)), key=lambda d: (fam_of(d), g_of(d) or 0, d))
    ctrls = {c: sorted((d for d in common if re.fullmatch(rf"{c}_\d+", d)), key=lambda x: int(x.split("_")[-1]))
             for c in set(CONTROL.values())}
    P(f"  directions at every start: {dirs}\n  controls: {ctrls}")
    starts = sorted({r["start"] for r in rows["A"] if r["start"] != "untrained"})
    golds = [s for s in starts if s.startswith("gold_mf_")]
    Gs = sorted({g_of(d) for d in dirs if fam_of(d) == "raw"})
    result = {"dirs": dirs, "golds": golds, "Gs": Gs}
    frozen = {}
    for dn in [d for d in dirs if fam_of(d) in frozen_fams] + [d for d in dirs if d == "adam_run2"]:
        P(f"\n## frozen run_part1 with v0 := {dn}, random := {CONTROL[fam_of(dn)]}_* (B={B})")
        with capture_frozen() as buf:
            S15.run_part1(rows_for(rows, dn, CONTROL[fam_of(dn)]), covs, B, c_phi=c_phi)
        frozen[dn] = parse_frozen(buf)
    for g, gname in (("correct_lenient", "len"), ("correct_marked", "mk")):
        arrs = S15.p1_arrays(rows, covs, g)
        N = len(arrs[0])
        fn = m3_stats_fn(arrs, golds, dirs, ctrls)
        obs = fn(np.arange(N), [1] * len(golds))
        rng = np.random.default_rng(seed)
        reps = defaultdict(list)
        for _ in range(B):
            J = rng.integers(0, N, N)
            w = np.bincount(rng.integers(0, len(golds), len(golds)), minlength=len(golds)).tolist()
            for k, x in fn(J, w).items():
                reps[k].append(x)
        P(f"\n## [{gname}] joint statistics (items {N}; gold starts {golds}; B={B})")
        for k in sorted(obs):
            P(f"  {k:46s} {fmt(obs[k], L.ci(reps[k]))}")
        if check_frozen_values and gname == "len":
            diffs = [abs(obs[f"K1|{st}|{dn}"] - fz[f"K1|{st}|v0"]) for dn, fz in frozen.items() for st in golds
                     if f"K1|{st}|v0" in fz]
            P(f"  cross-check vs frozen run_part1 (K1|start|v0, 3 dp): max |diff| {max(diffs) if diffs else np.nan:.4f} "
              f"over {len(diffs)} values")
        table = {dn: [k1_pass(obs, st, dn) for st in golds] for dn in dirs}
        P(f"  K1' passes (frozen thresholds; matched controls; same-direction placebo at theta_0):")
        for fam in FAMILIES + ("adam_run",):
            ds = [d for d in dirs if fam_of(d) == fam]
            if ds:
                P(f"    {fam:9s} " + "  ".join(f"{d.split('_')[-1]}: {sum(table[d])}/{len(golds)}" for d in ds))
        prim = f"{REG19['ADAM_PRIMARY']}_g{REG19['G_PRIMARY']}"
        adam_tab = {d: [adam_pass(obs, st, d) for st in golds] for d in dirs if fam_of(d) in ("sign", "adamv", "adamv25")}
        P(f"  Adam rule (K1' AND K1 <= K1(raw, same G) - {REG19['D_RAW']}): " +
          "  ".join(f"{d}: {sum(v)}/{len(golds)}" for d, v in adam_tab.items()))
        m3a = prim in adam_tab and sum(adam_tab[prim]) >= REG19["MIN_PASS"]
        P(f"  M3-A [{gname}] {prim} beats the raw direction, its matched control and its placebo at >= "
          f"{REG19['MIN_PASS']} of {len(golds)} gold starts: " + ("HOLDS" if m3a else
                                                                 ("FAILS" if prim in adam_tab else "NOT SCORABLE")))
        m31 = all(sum(table.get(f"raw_g{G}", [])) >= REG19["MIN_PASS"] for G in REG19["G_PASS"] if f"raw_g{G}" in table)
        P(f"  M3-1 [{gname}] raw K1' at >= {REG19['MIN_PASS']} starts for every G in {REG19['G_PASS']}: "
          + ("HOLDS" if m31 else "FAILS"))
        ci_slope = L.ci(reps.get("trend_slope_log2G|raw", []))
        m32 = bool(ci_slope[1] < 0)
        P(f"  M3-2 [{gname}] slope of mean raw K1 on log2 G {fmt(obs.get('trend_slope_log2G|raw', np.nan), ci_slope, 4)} -> "
          + ("HOLDS" if m32 else "FAILS"))
        any_pass = any(sum(v) >= REG19["MIN_PASS"] for v in table.values())
        P(f"  reading [{gname}]: " + ("some first-order direction predicts (see table)" if any_pass else
                                      "no raw or optimiser-geometry direction predicts at any G -> reversion is a "
                                      "multi-step effect (registered reading)"))
        result[gname] = {"obs": obs, "ci": {k: L.ci(v) for k, v in reps.items()}, "k1_table": table,
                         "adam_table": adam_tab, "M3_A": m3a, "M3_1": m31, "M3_2": m32, "any": any_pass}
    if grams:
        P("\n## raw-gradient SNR (per-batch Gram; batch bootstrap) and weight-space cosines")
        res = m3_gradient_report(grams, Gs, B=B)
        print_gradient_report(res, Gs)
        result["grad"] = res
        well = [st for st in golds if st in res and res[st]["snr1_W"][0] * max(Gs) >= REG19["SNR_WELL"]]
        P(f"  M3-3 guard: gold starts whose G={max(Gs)} raw mean is well estimated (SNR >= {REG19['SNR_WELL']}): "
          f"{len(well)}/{len(golds)}")
    if adam:
        P("\n## weight-space cosines of all directions with v25 / v150 / Part 1's v0 / SFT (point estimates)")
        print_adam_cosines(adam, grams)
    return result


# =========================================================================================================== M4
def fold(Y, d, d_new, nnz):
    m, b_new = d // d_new, d_new // nnz
    return Y.reshape(*Y.shape[:-1], nnz, m, b_new).sum(-2).reshape(*Y.shape[:-1], d_new)


def ga_load(gdir, dtag, name, temp):
    d = Path(gdir) / dtag / f"{name}_T{temp:g}"
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    flags = {}
    for r in L.read_jsonl(d / "samples.jsonl"):
        flags[(int(r["item"]), int(r["sample"]))] = (bool(r["correct_lenient"]), bool(r["correct_marked"]),
                                                    bool(r["stopped"]), int(r["n_tokens"]))
    it, gr, sk = [], [], []
    for f in sorted(d.glob("grad_*.safetensors")):
        t = st_load(f)
        it.append(t["items"])
        gr.append(t["gram"])
        sk.append(t["sketch"])
    if not it:
        return None
    items = np.concatenate(it)
    o = np.argsort(items, kind="stable")
    items, gram, sketch = items[o], np.concatenate(gr)[o], np.concatenate(sk)[o]
    assert len(np.unique(items)) == len(items), f"{d}: duplicate items across shards"
    K = meta["k"]
    arr = np.array([[flags[(int(e), s)] for s in range(K)] for e in items])  # [n, K, 4]
    return {"meta": meta, "items": items, "gram": gram, "sketch": sketch, "y": {"len": arr[..., 0].astype(bool),
            "mk": arr[..., 1].astype(bool)}, "stop": arr[..., 2].astype(bool), "ntok": arr[..., 3]}


def sf_coefs(y, mask):
    m = mask[None, :]
    n1, n0 = (y & m).sum(1), (~y & m).sum(1)
    c = np.where(y & m, 1.0 / np.maximum(n1, 1)[:, None], 0.0) - np.where(~y & m, 1.0 / np.maximum(n0, 1)[:, None], 0.0)
    return c, (n1 > 0) & (n0 > 0)


def ga_aggregate(ga, grader, dims, nnz):
    y = ga["y"][grader]
    K = y.shape[1]
    h1 = np.arange(K) < K // 2
    cF, vF = sf_coefs(y, np.ones(K, bool))
    c1, v1 = sf_coefs(y, h1)
    c2, v2 = sf_coefs(y, ~h1)
    G = ga["gram"]
    dmax = ga["sketch"].shape[-1]
    sk = ga["sketch"]  # float32: the aggregate is formed in float32 (relative error ~1e-7), then held in float64
    YF, Y1, Y2 = (np.einsum("nk,nkd->nd", c.astype(np.float32), sk).astype(np.float64) for c in (cF, c1, c2))
    return {"items": ga["items"], "p": y.mean(1), "validF": vF, "validX": v1 & v2,
            "denF": np.einsum("nk,nkl,nl->n", cF, G, cF), "denX": np.einsum("nk,nkl,nl->n", c1, G, c2),
            "Y": {d: tuple(fold(Y, dmax, d, nnz) for Y in (YF, Y1, Y2)) for d in dims if d <= dmax and dmax % d == 0}}


def pool_basis(agg, mask, d, weight):
    valid = agg["validF"] & mask
    w = np.sqrt(agg["p"] * (1 - agg["p"])) if weight == "grpo" else np.ones_like(agg["p"])
    M = agg["Y"][d][0][valid] * w[valid][:, None]
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt, S, int(valid.sum())


def align_items(agg, Vt, d, r):
    k = Vt.shape[0] if r == 0 else min(r, Vt.shape[0])
    W = Vt[:k]
    YF, Y1, Y2 = agg["Y"][d]
    f = k / d
    if f >= 0.95:  # the subspace fills the sketch: alignment is not identifiable at this d
        nan = np.full(len(YF), np.nan)
        return {"a2": nan, "num": nan.copy(), "den": nan.copy(), "k": k}
    ZF, Z1, Z2 = YF @ W.T, Y1 @ W.T, Y2 @ W.T
    with np.errstate(divide="ignore", invalid="ignore"):
        a2 = ((ZF ** 2).sum(1) / agg["denF"] - f) / (1 - f)
    num = ((Z1 * Z2).sum(1) - f * agg["denX"]) / (1 - f)
    return {"a2": np.where(agg["validF"], a2, np.nan), "num": np.where(agg["validX"], num, np.nan),
            "den": np.where(agg["validX"], agg["denX"], np.nan), "k": k}


def set_level(al, J):
    num, den = al["num"][J], al["den"][J]
    ok = np.isfinite(num) & np.isfinite(den)
    return float(num[ok].sum() / den[ok].sum()) if ok.any() and den[ok].sum() > 0 else np.nan


def ratio_sqrt(a, b):
    return math.sqrt(a / b) if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0 else np.nan


def dd_design(paths):
    """prereg 16's cells in its primary 'paired' composition (part_16's code path, frozen functions only)."""
    ev_A, unt_A, e2u = S15.load_A(paths)
    bad_train, _ = S15.part_A_exclusions(paths)
    data = S15.load_16(paths)
    per_set_bad = defaultdict(set)
    all_sets = {"gsm8k": {**ev_A, ("untrained", None, "pre"): unt_A}, **data}
    for st, evs in all_sets.items():
        for (arm, s, stage), e in evs.items():
            if arm != "untrained" and e.a["stop"].mean() < S15.REG["STOP_MIN"]:
                per_set_bad[st].add((arm, s))
    paired_bad = set().union(*per_set_bad.values()) if per_set_bad else set()
    gp_unt = data["gsm_plus"][("untrained", None, "pre")]
    clusters, seed_mask, ptypes, has_seed = S15.gsm_plus_clusters(unt_A.ids, unt_A.gold, gp_unt.ids, gp_unt.gold)
    bad = bad_train | paired_bad
    cells = dict(S15.cells_A(ev_A, unt_A, e2u, bad, steps=(375,), prefix="gsm8k:"))
    sets_N, ids = {"gsm8k": len(unt_A.ids)}, {"gsm8k": list(unt_A.ids)}
    for st, evs in data.items():
        cells[f"{st}:U"] = L.Cell.from_evals(f"{st}:U", st, None, [(0, [evs[("untrained", None, "pre")]])])
        for arm in S15.ARMS_A:
            ss = [s for s in SEEDS if (arm, s) not in bad]
            for stage, tag in (("pre", "pre"), (375, "375")):
                cells[f"{st}:{arm}|{tag}"] = L.Cell.from_evals(f"{st}:{arm}|{tag}", st, arm,
                                                               [(s, [evs[(arm, s, stage)]]) for s in ss])
        ss = [s for s in SEEDS if ("grpo", s) not in bad]
        cells[f"{st}:grpo|375"] = L.Cell.from_evals(f"{st}:grpo|375", st, "grpo", [(s, [evs[("grpo", s, 375)]]) for s in ss])
        sets_N[st] = len(evs[("untrained", None, "pre")].ids)
        ids[st] = list(evs[("untrained", None, "pre")].ids)
    P(f"  Delta D cells: paired exclusions {sorted(paired_bad | bad_train)}; items {sets_N}")
    return S15.build_16_design(cells, sets_N, clusters, seed_mask, has_seed), ids


ARM_GROUPS = {"gold_250": ("gold_250",), "gold_pool": ("gold_250", "gold_500"), "teacher_pool": ("teacher_250", "teacher_500")}


def delta_D(v, st, arms, o):
    full, h0, h1 = [], [], []
    U = v.m(f"{st}:U", o)
    QR = v.m(f"{st}:grpo|375", o)
    for a in arms:
        S, QS = v.m(f"{st}:{a}|pre", o), v.m(f"{st}:{a}|375", o)
        full.append((QS[2] - QR[2]) - (S[2] - U[2]))
        if all(x is not None for x in (S[0], S[1], QS[0], QS[1], QR[0], QR[1], U[0], U[1])):
            h0.append((QS[0] - QR[0]) - (S[0] - U[0]))
            h1.append((QS[1] - QR[1]) - (S[1] - U[1]))
    m = lambda xs: np.mean(xs, 0) if xs else None
    return m(full), m(h0), m(h1)


def tercile_contrast(a, prod, min_n=30):
    """mean(prod | top third of a) - mean(prod | bottom third); prod = Delta D^(1) Delta D^(2) is unbiased for Delta D^2."""
    ok = np.isfinite(a) & np.isfinite(prod)
    if ok.sum() < min_n:
        return np.nan
    a, prod = a[ok], prod[ok]
    lo, hi = np.quantile(a, [1 / 3, 2 / 3])
    return float(prod[a >= hi].mean() - prod[a <= lo].mean())


def tercile_contrast_strata(a, prod, strata, min_n=9):
    """The tercile contrast within each stratum (the own start's galign count n1), averaged with stratum-size weights."""
    ok = np.isfinite(a) & np.isfinite(prod) & np.isfinite(strata)
    num = den = 0.0
    for s in np.unique(strata[ok]):
        m = ok & (strata == s)
        t = tercile_contrast(a[m], prod[m], min_n)
        if np.isfinite(t):
            num += m.sum() * t
            den += m.sum()
    return num / den if den else np.nan


def strata_dummies(n_own, n_other, K=16):
    """Covariate columns for (b): one dummy per own-start galign count n1 = 2..K-1 (n1 sets the sampling noise of g_SF,
    hence the attenuation of a_naive; n1 = 0, K have no g_SF; n1 = 1 is the reference level) and the other start's count
    in 5 bins (0, 1-5, 6-10, 11-15, 16). NaN where a count is missing."""
    miss = ~np.isfinite(n_own) | ~np.isfinite(n_other)
    cols = [np.where(miss, np.nan, (n_own == c).astype(float)) for c in range(2, K)]
    ob = np.digitize(np.nan_to_num(n_other), [1, 6, 11, K])
    cols += [np.where(miss, np.nan, (ob == b).astype(float)) for b in range(1, 5)]
    return cols


def ga_load_all(gdir, temp):
    """{start kind: {set: galign arrays}} for the M4 starts that have pool gradients (loaded once)."""
    out = {}
    for dtag, kind in M4_STARTS.items():
        gas = {s: ga_load(gdir, dtag, s, temp) for s in GA_SETS + ("pool",)}
        if gas["pool"] is None:
            P(f"  [m4] {dtag}: no pool gradients, skipped")
            continue
        out[kind] = {s: g for s, g in gas.items() if g is not None}
    return out


def m4_item_arrays(loaded, grader, dims, r_list, weight, partition_path, nnz=None):
    """Per start: per set per (d, r) item arrays in example_id order; pool items cross-fitted (reported)."""
    out = {}
    for kind, gas in loaded.items():
        nnz_ = nnz or gas["pool"]["meta"]["sketch"]["nnz"]
        aggs = {s: ga_aggregate(g, grader, dims, nnz_) for s, g in gas.items()}
        pool_ids = set(json.loads(Path(partition_path).read_text())["grpo_example_ids"]) if kind == "gold" else None
        pmask = np.array([pool_ids is None or int(e) in pool_ids for e in aggs["pool"]["items"]])
        res = {"sets": {}, "p": {s: a["p"] for s, a in aggs.items()}, "items": {s: a["items"] for s, a in aggs.items()},
               "pool_n": {}, "spectrum": {}}
        par = aggs["pool"]["items"] % 2
        for d in dims:
            if d not in aggs["pool"]["Y"]:
                continue
            Vt, S, nval = pool_basis(aggs["pool"], pmask, d, weight)
            res["pool_n"][d] = nval
            res["spectrum"][d] = (S ** 2 / (S ** 2).sum())[:10].tolist()
            Vf = {f: pool_basis(aggs["pool"], pmask & (par == f), d, weight)[0] for f in (0, 1)}
            for r in r_list:
                for s, a in aggs.items():
                    if s != "pool":
                        res["sets"][(s, d, r)] = align_items(a, Vt, d, r)
                # pool items, cross-fitted by prompt folds (id parity): P from the other fold
                al = {"a2": np.full(len(par), np.nan), "num": np.full(len(par), np.nan), "den": np.full(len(par), np.nan)}
                for f in (0, 1):
                    af = align_items(aggs["pool"], Vf[f], d, r)
                    for k in ("a2", "num", "den"):
                        al[k][(par == 1 - f) & pmask] = af[k][(par == 1 - f) & pmask]
                res["sets"][("pool_xfit", d, r)] = al
        out[kind] = res
    return out


def to_cell_order(arr, items, ids):
    pos = {int(e): j for j, e in enumerate(items)}
    return np.array([arr[pos[int(e)]] if int(e) in pos else np.nan for e in ids], float)


def score_m4(root, gdir, B, temp=REG19["TEMP"], dims=REG19["D_LIST"], r_list=REG19["R_LIST"], d0=REG19["D"],
             r0=REG19["R"], weight=REG19["POOL_WEIGHT"], partition=None, seed=1912):
    P(f"\n# M4 — gradient alignment vs dataset distance (galign {gdir}, T {temp}); primary d={d0}, r={r0}, "
      f"pool weight {weight}")
    paths = S15.Paths(root, "runs/gsm8k/Llama-3.2-1B-Instruct-px-{proto}{src}")
    design, ids = dd_design(paths)
    partition = partition or str(Path(root) / GOLD_PARTITION)
    result = {}
    loaded = ga_load_all(gdir, temp)
    for grader in ("len", "mk"):
        arrays = m4_item_arrays(loaded, grader, dims, r_list, weight, partition)
        if not arrays:
            P("  no galign outputs: M4 not scored")
            return {}
        # covariates for (b): p_U and p_S on the galign samples (independent of Delta D's samples), and the galign counts
        # n1 = K p (they set the sampling noise of g_SF, hence a_naive's attenuation)
        cov = {s: [to_cell_order(arrays[k]["p"][s], arrays[k]["items"][s], ids[s]) for k in ("untrained", "gold")
                   if k in arrays and s in arrays[k]["p"]] for s in GA_SETS}
        Kga = next(iter(loaded.values()))["pool"]["meta"]["k"]
        cnt = {k: {s: np.round(Kga * to_cell_order(arrays[k]["p"][s], arrays[k]["items"][s], ids[s]))
                   for s in GA_SETS if s in arrays[k]["p"]} for k in arrays}
        other = {"gold": "untrained", "untrained": "gold"}
        A = {}
        for kind, res in arrays.items():
            for (s, d, r), al in res["sets"].items():
                if s in GA_SETS:
                    A[(kind, s, d, r)] = {k: to_cell_order(al[k], res["items"][s], ids[s]) for k in ("a2", "num", "den")}
        P(f"\n## [{grader}] pool rows used {[(k, v['pool_n']) for k, v in arrays.items()]}; top-10 pool energy shares "
          f"(d={d0}): " + "; ".join(f"{k}: " + " ".join(f"{x:.3f}" for x in v["spectrum"].get(d0, [])) for k, v in arrays.items()))
        for kind, res in arrays.items():
            for s in GA_SETS:
                al = A.get((kind, s, d0, r0))
                if al is not None:
                    P(f"  {kind:9s} {s:9s} items with g_SF: all-sample {np.isfinite(al['a2']).sum()}, both halves "
                      f"{np.isfinite(al['num']).sum()} of {len(ids[s])}")

        def fn(v):
            out = {}
            J = {s: v.idx(s) for s in GA_SETS}
            for (kind, s, d, r), al in A.items():
                out[f"A2|{kind}|{s}|d{d}|r{r}"] = set_level(al, J[s])
                a = np.sqrt(np.clip(al["a2"][J[s]], 0, None))
                out[f"naive|{kind}|{s}|d{d}|r{r}"] = float(np.nanmean(a))
            for kind in arrays:
                for d in dims:
                    for r in r_list:
                        if (kind, "gsm8k", d, r) not in A:
                            continue
                        num = sum(np.nansum(A[(kind, s, d, r)]["num"][J[s]]) for s in OOD)
                        den = sum(np.nansum(np.where(np.isfinite(A[(kind, s, d, r)]["num"][J[s]]),
                                                     A[(kind, s, d, r)]["den"][J[s]], np.nan)) for s in OOD)
                        a_off = num / den if den > 0 else np.nan
                        out[f"A2|{kind}|ood|d{d}|r{r}"] = a_off
                        out[f"ratio|{kind}|d{d}|r{r}"] = ratio_sqrt(a_off, out[f"A2|{kind}|gsm8k|d{d}|r{r}"])
                        for s in OOD:
                            out[f"ratio|{kind}|{s}|d{d}|r{r}"] = ratio_sqrt(out[f"A2|{kind}|{s}|d{d}|r{r}"],
                                                                            out[f"A2|{kind}|gsm8k|d{d}|r{r}"])
                        nv = [x for s in OOD for x in np.sqrt(np.clip(A[(kind, s, d, r)]["a2"][J[s]], 0, None))]
                        out[f"ratio_naive|{kind}|d{d}|r{r}"] = float(np.nanmean(nv) / out[f"naive|{kind}|gsm8k|d{d}|r{r}"])
            # (b) at the primary (d, r); pooled = item-weighted mean of the per-set stratified tercile contrasts
            pool_b = defaultdict(list)
            for s in GA_SETS:
                Z = [c[J[s]] for c in cov[s]]
                for ag, arms in ARM_GROUPS.items():
                    full, h0, h1 = delta_D(v, s, arms, grader)
                    absd = np.abs(full)
                    prod = h0 * h1 if h0 is not None else None
                    for kind in arrays:
                        al = A.get((kind, s, d0, r0))
                        if al is None:
                            continue
                        a = np.sqrt(np.clip(al["a2"][J[s]], 0, None))
                        key = f"{kind}|{s}|{ag}"
                        own = cnt[kind][s][J[s]]
                        oth = cnt[other[kind]][s][J[s]] if other[kind] in cnt and s in cnt[other[kind]] else own
                        Zs = strata_dummies(own, oth, Kga)
                        out[f"b_partial|{key}"] = L.pspear(absd, a, Z) if Z else np.nan
                        out[f"b_strata|{key}"] = L.pspear(absd, a, Zs)
                        out[f"b_plain|{key}"] = spear(a, absd)
                        out[f"b_pnorm|{key}"] = L.pspear(absd, al["num"][J[s]], Zs)
                        out[f"b_tercile|{key}"] = tercile_contrast(a, prod) if prod is not None else np.nan
                        out[f"b_tercile_strata|{key}"] = (tercile_contrast_strata(a, prod, own)
                                                          if prod is not None else np.nan)
                        if prod is not None:
                            n_ok = int((np.isfinite(a) & np.isfinite(prod)).sum())
                            pool_b[(kind, ag)].append((out[f"b_tercile_strata|{key}"], n_ok))
            for (kind, ag), lst in pool_b.items():
                ok = [(t, n) for t, n in lst if np.isfinite(t) and n > 0]
                out[f"b_tercile_strata|{kind}|pooled|{ag}"] = (sum(t * n for t, n in ok) / sum(n for _, n in ok)
                                                               if ok else np.nan)
            return out
        obs, reps = L.bootstrap(design, fn, B, seed=seed)
        _, reps_i = L.bootstrap(design, fn, max(B // 4, 200), seed=seed + 1, seeds=False)
        P(f"\n## [{grader}] statistics (95% CI: hierarchical, B={B}; item-only CI in braces for (b), B={max(B // 4, 200)})")
        for k in sorted(obs):
            extra = f"  {{{L.ci(reps_i[k])[0]:+.3f}, {L.ci(reps_i[k])[1]:+.3f}}}" if k.startswith("b_") else ""
            P(f"  {k:44s} {fmt(obs[k], L.ci(reps[k]))}{extra}")
        # pool items, cross-fitted (reported)
        for kind, res in arrays.items():
            al = res["sets"].get(("pool_xfit", d0, r0))
            if al is not None:
                A2p = set_level(al, np.arange(len(al["num"])))
                P(f"  pool items (P from the other id-parity fold) {kind}: A = {math.sqrt(A2p) if A2p > 0 else np.nan:.4f}")
        verdict = {}
        for kind in arrays:
            k = f"ratio|{kind}|d{d0}|r{r0}"
            lo, hi = L.ci(reps[k])
            va = ("HOLDS (dataset distance does not separate alignment)" if lo >= REG19["RATIO_MIN"] else
                  "FAILS (off-distribution alignment is lower)" if hi < REG19["RATIO_MIN"] else "INCONCLUSIVE")
            P(f"  M4-a [{grader}] {kind} start: A_off / A_in {fmt(obs[k], (lo, hi))} vs {REG19['RATIO_MIN']} -> {va}")
            nb = {s: bool(L.ci(reps[f"{REG19['B_STAT']}|{kind}|{s}|{REG19['ARM']}"])[0] > 0) for s in GA_SETS + ("pooled",)
                  if f"{REG19['B_STAT']}|{kind}|{s}|{REG19['ARM']}" in reps}
            kp = f"{REG19['B_STAT']}|{kind}|pooled|{REG19['ARM']}"
            P(f"  M4-b [{grader}] {kind} start ({REG19['ARM']}): {REG19['B_STAT']} pooled over the 4 sets "
              f"{fmt(obs.get(kp, np.nan), L.ci(reps.get(kp, [])), 4)} -> " + ("HOLDS" if nb.get("pooled") else "FAILS")
              + f"; CI above 0 per set (reported): {{{', '.join(f'{s}: {v}' for s, v in nb.items() if s != 'pooled')}}}")
            verdict[kind] = {"a": va, "b": nb}
        result[grader] = {"obs": obs, "ci": {k: L.ci(v) for k, v in reps.items()}, "verdict": verdict}
    return result


def export_p1(p1_reg, m3_dir, out_base):
    """Format conversion only (no statistics): <out_base>_<direction>/{A,B,C,covariates}.jsonl in the frozen P1_DOC
    format (v0 := the direction, rand_<j> := its matched controls), so the unmodified frozen CLI can be run per direction:
      python -I prereg/scripts/p15/score_prereg15.py --part 1 --unblind --p1_dir <out_base>_<direction> --c_phi 0.37"""
    rows, covs, _, _, found = m3_load(p1_reg, m3_dir)
    common = set.intersection(*(set(v) for v in found.values())) if found else set()
    for dn in sorted(d for d in common if fam_of(d)):
        rg = rows_for(rows, dn, CONTROL[fam_of(dn)])
        out = Path(f"{out_base}_{dn}")
        out.mkdir(parents=True, exist_ok=True)
        for s in "ABC":
            (out / f"{s}.jsonl").write_text("".join(json.dumps({**r, "set": s}) + "\n" for r in rg[s]))
        (out / "covariates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in covs))
        P(f"  export: wrote {out} ({len(rg['A'])} A rows)")


# =========================================================================================================== schema
def schema(root, p1_reg, m3_dir, gdir, temp):
    P("\n# schema only (no correctness, no phi values, no gradient values are read into statistics)")
    p1_reg, m3_dir, gdir = Path(root) / p1_reg, Path(root) / m3_dir, Path(root) / gdir
    for f in ("A", "B", "C", "covariates"):
        p = p1_reg / f"{f}.jsonl"
        if p.exists():
            keys, n, starts, dirs = set(), 0, set(), set()
            for r in L.read_jsonl(p):
                keys |= set(r)
                n += 1
                starts.add(r.get("start"))
                dirs.add(r.get("direction_name"))
            P(f"  {p}: rows {n}; starts {sorted(map(str, starts))}; directions {sorted(map(str, dirs))}; keys {sorted(keys)}")
    for dtag in M3_STARTS:
        p = m3_dir / f"phix_{dtag}_A.jsonl"
        if p.exists():
            cnt = defaultdict(int)
            for r in L.read_jsonl(p):
                cnt[r["direction_name"]] += 1
            P(f"  {p}: {dict(cnt)}")
        g = m3_dir / "directions" / dtag / "m3_gram.json"
        if g.exists():
            meta = json.loads(g.read_text())
            P(f"  {g}: n_batches {meta['n_batches']}; refs {meta['refs']}; pool {meta['pool_size']}")
    for dtag in M4_STARTS:
        for s in GA_SETS + ("pool",):
            d = gdir / dtag / f"{s}_T{temp:g}"
            if not d.exists():
                P(f"  {d}: missing")
                continue
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            n_rows = sum(1 for _ in L.read_jsonl(d / "samples.jsonl")) if (d / "samples.jsonl").exists() else 0
            shards = sorted(d.glob("grad_*.safetensors"))
            n_items = 0
            for f in shards:
                from safetensors import safe_open
                with safe_open(str(f), "numpy") as h:
                    n_items += h.get_slice("items").get_shape()[0]
            P(f"  {d}: samples {n_rows}; shards {len(shards)}; items with gradients {n_items}; K {meta.get('k')}; "
              f"sketch {meta.get('sketch', {}).get('d')}/{meta.get('sketch', {}).get('digest')}; D {meta.get('D')}")


# =========================================================================================================== dry runs
def osnap_np(D, d, nnz, rng):
    b = d // nnz
    idx = np.stack([k * b + rng.integers(0, b, D) for k in range(nnz)])
    sgn = np.stack([(rng.integers(0, 2, D) * 2 - 1) / math.sqrt(nnz) for _ in range(nnz)])
    return idx, sgn


def apply_osnap(X, idx, sgn, d):
    Y = np.zeros(X.shape[:-1] + (d,))
    Xf = X.reshape(-1, X.shape[-1])
    Yf = Y.reshape(-1, d)
    for k in range(idx.shape[0]):
        for i in range(Xf.shape[0]):
            Yf[i] += np.bincount(idx[k], weights=Xf[i] * sgn[k], minlength=d)
    return Yf.reshape(Y.shape)


def synth_ga(rng, n, K, D, Q, a_mean, p_ab, sketch, noise_out=3.0, noise_in=0.5, base=2.0, a_sd=0.15):
    """Synthetic galign arrays: g(x) = |g| (a u_in + sqrt(1-a^2) u_out), u_in in span(Q), u_out orthogonal;
    per-sample g_i = b(x) + y_i g(x) + e_i with e_i = noise_in Q xi + noise_out eta (in units of |g| = 1)."""
    m = Q.shape[1]
    a = np.clip(rng.normal(a_mean, a_sd, n), 0.02, 0.98)
    p = rng.beta(*p_ab, n)
    y = rng.random((n, K)) < p[:, None]
    X = np.zeros((n, K, D))
    for i in range(n):
        uin = Q @ rng.normal(size=m)
        uin /= np.linalg.norm(uin)
        uo = rng.normal(size=D)
        uo -= Q @ (Q.T @ uo)
        uo /= np.linalg.norm(uo)
        g = a[i] * uin + math.sqrt(1 - a[i] ** 2) * uo
        b = base * rng.normal(size=D) / math.sqrt(D)
        e = noise_in * (rng.normal(size=(K, m)) @ Q.T) / math.sqrt(m) + noise_out * rng.normal(size=(K, D)) / math.sqrt(D)
        X[i] = b + y[i][:, None] * g + e
    gram = np.einsum("nkd,nld->nkl", X, X)
    idx, sgn, d = sketch
    sk = apply_osnap(X, idx, sgn, d).astype(np.float32)
    return {"items": np.arange(n), "gram": gram, "sketch": sk, "y": {"len": y, "mk": y}, "a": a, "p": p}


def dry_snr(rng, reps=200, B=300):
    P("\n## S1. M3 SNR estimator (U-statistic; batch bootstrap): bias and 95% CI coverage over synthetic Gram matrices")
    for s1, nb in ((0.05, 64), (0.2, 64), (1.0, 64), (0.2, 16)):
        D = 2000
        est, cov = [], 0
        for _ in range(reps):
            mu = rng.normal(size=D)
            mu *= math.sqrt(s1) / np.linalg.norm(mu)
            Gb = mu + rng.normal(size=(nb, D)) / math.sqrt(D)  # tr Sigma = 1
            K = Gb @ Gb.T
            e = gram_snr(K, np.arange(nb))[2]
            bs = np.array([gram_snr(K, rng.integers(0, nb, nb))[2] for _ in range(B)])
            lo, hi = L.ci(bs)
            est.append(e)
            cov += lo <= s1 <= hi
        P(f"  true SNR per batch {s1:5.2f}, {nb} batches: mean estimate {np.mean(est):.3f} (sd {np.std(est):.3f}); "
          f"CI coverage {cov / reps:.2f}")


def synth_p1_rows(rng, world, Gs, n_rand=4, N=600, K=16, fams=("raw", "adamv"), rho=0.5):
    """Synthetic Part 1 + M3 rows in the scorer's names (raw_g<G>, adamv_g<G>, craw_<j>, cadamv_<j>, v25, v150).
    phi along fam_g<G> = e_fam + 1[correct] (gain_fam m(x) + noise_fam,G(x)); noise_fam,G = nested mean of per-batch item
    effects (sd sz / sqrt(G)); e_adamv = rho e_raw + sqrt(1 - rho^2) e'. Controls: 1[correct] (0.3 xi_j std(D_SFT) +
    N(0, 0.6^2)) + N(0, 1). world: null (no signal), noisy (signal, large batch noise), strong (signal, small noise),
    adam (Adam direction carries the signal, raw only 30%)."""
    strength, sz, gain = {"null": (0.0, 1.0, {"raw": 1, "adamv": 1}), "noisy": (1.0, 3.0, {"raw": 1, "adamv": 1}),
                          "strong": (1.0, 0.5, {"raw": 1, "adamv": 1}), "adam": (1.0, 1.0, {"raw": 0.3, "adamv": 1})}[world]
    th = rng.normal(0, 2.0, N)
    rev = rng.normal(0, 1, N)
    rows = {"A": [], "B": [], "C": []}
    p0 = L.expit(th)
    Gmax = max(Gs)
    for st in ("untrained", "gold_mf_s42", "gold_mf_s43", "gold_mf_s44"):
        dS = 0 if st == "untrained" else rng.normal(-0.5, 1.0, N)
        p = L.expit(th + dS)
        dsft = p - p0
        mech = strength * (-3 * dsft + 0.5 * rev) if st != "untrained" else 0.1 * th
        zb = {f: rng.normal(0, sz, (Gmax, N)).cumsum(0) / np.arange(1, Gmax + 1)[:, None] for f in fams}
        sd = (dsft - dsft.mean()) / (dsft.std() + 1e-12)
        rnd = {f: [0.3 * rng.choice([-1, 1]) * sd + rng.normal(0, 0.6, N) for _ in range(n_rand)] for f in fams}
        for s_ in "AB":
            for i in range(N):
                for k in range(K):
                    c = bool(rng.random() < p[i])
                    base = {"item": i, "sample": k, "start": st, "correct_lenient": c, "correct_marked": c}
                    if s_ == "B":
                        rows["B"].append(base)
                        continue
                    e = rng.normal(0, 1)
                    ef = {"raw": e, "adamv": rho * e + math.sqrt(1 - rho ** 2) * rng.normal(0, 1)}
                    for f in fams:
                        for G in Gs:
                            rows["A"].append({**base, "direction_name": f"{f}_g{G}",
                                              "phi": ef[f] + ((gain[f] * mech[i] + zb[f][G - 1, i]) if c else 0.0)})
                        for j in range(n_rand):
                            rows["A"].append({**base, "direction_name": f"{CONTROL[f]}_{j + 1}",
                                              "phi": rng.normal(0, 1) + (rnd[f][j][i] if c else 0.0)})
                    f0 = e + (mech[i] if c else 0.0)
                    rows["A"].append({**base, "direction_name": "v150", "phi": 0.6 * f0 + rng.normal(0, 0.8)})
                    rows["A"].append({**base, "direction_name": "v25", "phi": 0.3 * f0 + rng.normal(0, 0.5)})
        q = L.expit(th + 1.0 + (dS - strength * (2 * dsft - 0.8 * rev) if st != "untrained" else 0))
        for name in ([f"post_untrained_s{s}" for s in SEEDS] if st == "untrained" else [f"post_{st}"]):
            for i in range(N):
                for k in range(K):
                    c = bool(rng.random() < q[i])
                    rows["C"].append({"item": i, "sample": k, "start": name, "correct_lenient": c, "correct_marked": c})
    return rows


def dry_m3_rules(rng, reps=10, B=200, Gs=(1, 4, 16, 64)):
    P(f"\n## S2. M3 rules on synthetic Part 1 data (rows -> frozen p1_arrays -> replica; {reps} datasets per world, "
      f"B={B}): P(M3-1), P(M3-2), P(M3-A at G={max(Gs)}), mean K1bar per G (raw | adamv)")
    golds = ["gold_mf_s42", "gold_mf_s43", "gold_mf_s44"]
    for world in ("null", "noisy", "strong", "adam"):
        h1 = h2 = ha = 0
        k1 = defaultdict(list)
        for r in range(reps):
            rows = synth_p1_rows(rng, world, Gs)
            arrs = S15.p1_arrays(rows, [], "correct_lenient")
            dirs = [f"{f}_g{G}" for f in ("raw", "adamv") for G in Gs]
            ctrls = {c: [f"{c}_{j}" for j in range(1, 5)] for c in ("craw", "cadamv")}
            fn = m3_stats_fn(arrs, golds, dirs, ctrls, refs=("v25", "v150"))
            N = len(arrs[0])
            obs = fn(np.arange(N), [1, 1, 1])
            slopes = []
            for _ in range(B):
                J = rng.integers(0, N, N)
                w = np.bincount(rng.integers(0, 3, 3), minlength=3).tolist()
                slopes.append(fn(J, w).get("trend_slope_log2G|raw", np.nan))
            h1 += all(sum(k1_pass(obs, st, f"raw_g{G}") for st in golds) >= REG19["MIN_PASS"] for G in REG19["G_PASS"])
            h2 += L.ci(slopes)[1] < 0
            ha += sum(adam_pass(obs, st, f"adamv_g{max(Gs)}") for st in golds) >= REG19["MIN_PASS"]
            for d in dirs:
                k1[d].append(obs[f"K1bar|{d}"])
            if r == 0 and world == "strong":  # the replica against the frozen run_part1, once
                with capture_frozen() as buf:
                    S15.run_part1(rows_for(rows, f"raw_g{max(Gs)}", "craw"), [], 50)
                fz = parse_frozen(buf)
                diffs = [abs(obs[f"K1|{st}|raw_g{max(Gs)}"] - fz[f"K1|{st}|v0"]) for st in golds]
                P(f"  cross-check (strong world) replica vs frozen run_part1: max |diff| {max(diffs):.4f} (3 dp print)")
        P(f"  world {world:6s}: P(M3-1) {h1 / reps:.2f}  P(M3-2) {h2 / reps:.2f}  P(M3-A) {ha / reps:.2f}  K1bar "
          + "  ".join(f"G{G} {np.mean(k1[f'raw_g{G}']):+.3f}|{np.mean(k1[f'adamv_g{G}']):+.3f}" for G in Gs))


def fast_world(rng, gains, rho, sz, N=600, K=16, n_ctrl=4, strength=1.0):
    """Vectorised synthetic world for the D_RAW calibration: per start the K1 statistic of raw, adam, controls and the
    untrained-start placebo, computed exactly as the scorer does (Delta = mean phi correct - incorrect; partial Spearman
    with D_SFT = p_B(start) - p_B(untrained) given p_A(start))."""
    th = rng.normal(0, 2.0, N)
    rev = rng.normal(0, 1, N)
    p0 = L.expit(th)

    def draw(p):
        return rng.random((N, K)) < p[:, None]

    def delta(phi, y):
        n1, n0 = y.sum(1), (~y).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (phi * y).sum(1) / n1 - (phi * ~y).sum(1) / n0
        return np.where((n1 > 0) & (n0 > 0), d, np.nan)

    def phis(y, mech):
        e1, e2 = rng.normal(size=(N, K)), rng.normal(size=(N, K))
        z1 = rng.normal(0, sz, N)
        z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.normal(0, sz, N)
        return {"raw": e1 + y * (gains["raw"] * mech + z1)[:, None],
                "adam": rho * e1 + math.sqrt(1 - rho ** 2) * e2 + y * (gains["adam"] * mech + z2)[:, None]}
    yAU, yBU = draw(p0), draw(p0)
    plac = {k: delta(v, yAU) for k, v in phis(yAU, 0.1 * th).items()}
    out = []
    for _ in range(3):
        dS = rng.normal(-0.5, 1.0, N)
        p = L.expit(th + dS)
        dsft = p - p0
        mech = strength * (-3 * dsft + 0.5 * rev)
        yA, yB = draw(p), draw(p)
        dh = yB.mean(1) - yBU.mean(1)
        pS = yA.mean(1)
        k1 = {k: L.pspear(delta(v, yA), dh, [pS]) for k, v in phis(yA, mech).items()}
        sd = (dsft - dsft.mean()) / dsft.std()
        ctrl = np.nanmean([L.pspear(delta(rng.normal(size=(N, K)) + yA * (0.3 * rng.choice([-1, 1]) * sd +
                                                                         rng.normal(0, 0.6, N))[:, None], yA), dh, [pS])
                           for _ in range(n_ctrl)])
        pl = {k: L.pspear(v, dh, [pS]) for k, v in plac.items()}
        out.append((k1, ctrl, pl))
    return out


def dry_adam_margin(rng, reps=200, Ds=(0.0, 0.02, 0.03, 0.05, 0.08, 0.10)):
    P(f"\n## S2b. Calibration of D_RAW (Adam rule: K1' AND K1(adam) <= K1(raw) - D at >= 2 of 3 starts); {reps} worlds "
      f"per configuration. Null = equal signal in raw and Adam directions; power = Adam signal, raw 30% of it.")
    res = {}
    for rho in (0.0, 0.5, 0.8):
        for sz in (0.5, 2.0):
            for lab, gains in (("null", {"raw": 1.0, "adam": 1.0}), ("power", {"raw": 0.3, "adam": 1.0})):
                hits = np.zeros(len(Ds))
                for _ in range(reps):
                    w = fast_world(rng, gains, rho, sz)
                    for i, D in enumerate(Ds):
                        n = sum(bool(k1["adam"] <= REG19["K1_MAX"] and k1["adam"] <= ctrl - REG19["K_MARGIN"]
                                     and k1["adam"] <= pl["adam"] - REG19["K_MARGIN"] and k1["adam"] <= k1["raw"] - D)
                                for k1, ctrl, pl in w)
                        hits[i] += n >= REG19["MIN_PASS"]
                res[(rho, sz, lab)] = hits / reps
                P(f"  rho {rho:.1f} sz {sz:.1f} {lab:5s}: " + "  ".join(f"D={D:.2f}: {h:.3f}" for D, h in zip(Ds, hits / reps)))
    ok = [D for i, D in enumerate(Ds) if all(v[i] <= 0.05 for (rho, sz, lab), v in res.items() if lab == "null")]
    P(f"  -> smallest D with P(rule | equal signal) <= 0.05 in every null configuration: {min(ok) if ok else 'none'}")


def dry_m4_b(rng, reps=60, B=200):
    P("\n## S4. M4 (b) calibration with the realistic noise model: a(x) is attenuated by the sampling noise of g_SF, which "
      "is set by the REALISED galign counts n1 (K=16 at the start; also the covariates); |Delta D| from halves with "
      "3 seeds x 2 samples per half (prereg 14/16-like). Rejection rate of 'CI above 0' (item bootstrap B=200), N=1319.")

    def world(planted, N=1319):
        th = rng.normal(0, 1.8, N)
        pU = L.expit(th)
        dpre = rng.normal(-0.3, 1.0, N)
        pS = L.expit(th + dpre)
        a_true = rng.beta(2, 3, N)
        retain = 1 - (planted * a_true + (1 - planted) * 0.6)
        qR, qS = L.expit(th + 1.0), L.expit(th + 1.0 + retain * dpre)
        b = lambda p: rng.binomial(6, p) / 6
        h = [(b(qS) - b(qR)) - (b(pS) - b(pU)) for _ in range(2)]
        nS, nU = rng.binomial(16, pS), rng.binomial(16, pU)
        noise2 = 2.0 * (1 / np.maximum(nS, 1) + 1 / np.maximum(16 - nS, 1))
        a2 = a_true ** 2 / (1 + noise2) + rng.normal(0, 0.02, N)
        a = np.where((nS > 0) & (nS < 16), np.sqrt(np.clip(a2, 0, None)), np.nan)
        return a, h, nU, nS

    res = defaultdict(lambda: defaultdict(int))
    for planted in (0.0, 1.0):
        for _ in range(reps):
            a, h, nU, nS = world(planted)
            absd = np.abs((h[0] + h[1]) / 2)
            prod = h[0] * h[1]
            Zs = strata_dummies(nS, nU)
            stats = {"plain": lambda J: spear(a[J], absd[J]),
                     "partial_lin": lambda J: L.pspear(absd[J], a[J], [nU[J] / 16, nS[J] / 16]),
                     "partial_strata": lambda J: L.pspear(absd[J], a[J], [z[J] for z in Zs]),
                     "tercile": lambda J: tercile_contrast(a[J], prod[J]),
                     "tercile_strata": lambda J: tercile_contrast_strata(a[J], prod[J], nS[J])}
            for name, f in stats.items():
                bs = [f(rng.integers(0, len(a), len(a))) for _ in range(B)]
                res[planted][name] += L.ci(bs)[0] > 0
        P(f"  {'planted (erasure grows with alignment)' if planted else 'null (no alignment effect)'}: "
          + "  ".join(f"{k} {v / reps:.2f}" for k, v in res[planted].items()))
    # the registered pooled statistic: four sets' worth of items (1,319 + 2,093 + 1,000 + 600 = 5,012) in one world
    for planted in (0.0, 1.0):
        hit = 0
        for _ in range(max(reps // 2, 10)):
            a, h, nU, nS = world(planted, N=5012)
            prod = h[0] * h[1]
            bs = [tercile_contrast_strata(a[J], prod[J], nS[J]) for J in
                  (rng.integers(0, len(a), len(a)) for _ in range(B))]
            hit += L.ci(bs)[0] > 0
        P(f"  pooled size N=5012, {'planted' if planted else 'null'}: tercile_strata {hit / max(reps // 2, 10):.2f}")


def dry_synthetic(args):
    rng = np.random.default_rng(args.seed)
    sec = args.sections
    if "1" in sec:
        dry_snr(rng)
    if "2" in sec:
        dry_m3_rules(rng, reps=max(5, args.reps // 2), B=args.dry_B)
        dry_adam_margin(rng, reps=args.reps * 10)
    if "3" in sec:
        dry_m4_alignment(rng)
    if "4" in sec:
        dry_m4_b(rng, reps=args.reps * 3)


def dry_m4_alignment(rng, reps=10):
    P("\n## S3. M4 alignment estimators on synthetic gradients (D=3000, 16-dim RL subspace, K=16, OSNAP d=1024 nnz=4):"
      " cross-fitted vs naive A_set and ratio; sketch debiasing (r = 16 = true span)")
    D, m, K, d, nnz = 3000, 16, 16, 1024, 4
    for a_in, a_off, p_in, p_off, lab in ((0.5, 0.5, (2, 2), (1, 4), "equal alignment, harder OOD"),
                                          (0.5, 0.3, (2, 2), (2, 2), "lower OOD alignment (true ratio 0.6)")):
        rows = []
        for _ in range(reps):
            Q = np.linalg.qr(rng.normal(size=(D, m)))[0]
            sk = (*osnap_np(D, d, nnz, rng), d)
            pool = synth_ga(rng, 150, K, D, Q, 0.9, (2, 2), sk, a_sd=0.05)
            gin = synth_ga(rng, 150, K, D, Q, a_in, p_in, sk)
            goff = synth_ga(rng, 150, K, D, Q, a_off, p_off, sk)
            ap, ai, ao = (ga_aggregate(g, "len", (256, 512, 1024), nnz) for g in (pool, gin, goff))
            row = {}
            for dd in (256, 512, 1024):
                Vt, _, _ = pool_basis(ap, np.ones(150, bool), dd, "grpo")
                li, lo = align_items(ai, Vt, dd, m), align_items(ao, Vt, dd, m)
                Ai, Ao = set_level(li, np.arange(150)), set_level(lo, np.arange(150))
                row[dd] = (ratio_sqrt(Ao, Ai), np.nanmean(np.sqrt(np.clip(lo["a2"], 0, None))) /
                           np.nanmean(np.sqrt(np.clip(li["a2"], 0, None))), math.sqrt(max(Ai, 0)))
            true_in = math.sqrt((gin["a"] ** 2).mean())
            true_ratio = math.sqrt((goff["a"] ** 2).mean()) / true_in
            rows.append((true_ratio, true_in, row))
        P(f"  {lab}: true ratio {np.mean([r[0] for r in rows]):.3f}; true A_in {np.mean([r[1] for r in rows]):.3f}")
        for dd in (256, 512, 1024):
            P(f"    d={dd:5d}: cross-fitted ratio {np.mean([r[2][dd][0] for r in rows]):.3f} (sd "
              f"{np.std([r[2][dd][0] for r in rows]):.3f}); naive ratio {np.mean([r[2][dd][1] for r in rows]):.3f}; "
              f"cross-fitted A_in {np.mean([r[2][dd][2] for r in rows]):.3f}")


def write_mock_galign(rng, root, gdir, nnz=4, d=1024, D=1500, m=12, K=16):
    """Synthetic galign outputs for both M4 starts and every set, with the mock root's item ids (null world: the
    alignment is unrelated to the mock Delta D)."""
    from safetensors.numpy import save_file
    paths = S15.Paths(root, "runs/gsm8k/Llama-3.2-1B-Instruct-px-{proto}{src}")
    _, ids = dd_design(paths)
    ids["pool"] = list(range(1000))
    part = Path(root) / GOLD_PARTITION
    if not part.exists():
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps({"grpo_example_ids": sorted(rng.choice(1000, 750, replace=False).tolist())}))
    Q = np.linalg.qr(rng.normal(size=(D, m)))[0]
    sk = (*osnap_np(D, d, nnz, rng), d)
    for dtag in M4_STARTS:
        for s, a_mean in (("pool", 0.9), ("gsm8k", 0.5), ("gsm_plus", 0.5), ("svamp", 0.45), ("math600", 0.4)):
            out = Path(gdir) / dtag / f"{s}_T{REG19['TEMP']:g}"
            out.mkdir(parents=True, exist_ok=True)
            n = len(ids[s])
            ga = synth_ga(rng, n, K, D, Q, a_mean, (2, 3), sk, a_sd=0.1)
            with open(out / "samples.jsonl", "w") as f:
                for i, e in enumerate(ids[s]):
                    for k in range(K):
                        f.write(json.dumps({"item": int(e), "sample": k, "correct_lenient": bool(ga["y"]["len"][i, k]),
                                            "correct_marked": bool(ga["y"]["len"][i, k]), "stopped": True,
                                            "n_tokens": 100}) + "\n")
            for c0 in range(0, n, 256):
                save_file({"items": np.array(ids[s][c0:c0 + 256], np.int64), "gram": ga["gram"][c0:c0 + 256],
                           "sketch": ga["sketch"][c0:c0 + 256], "logp": np.zeros((min(256, n - c0), K))},
                          str(out / f"grad_{c0 // 256:05d}.safetensors"))
            (out / "meta.json").write_text(json.dumps({"k": K, "sketch": {"d": d, "nnz": nnz, "digest": "synthetic"},
                                                       "D": D, "synthetic": True}))
    P(f"  mock galign files written to {gdir}")


def write_mock_m3(rng, root, Gs, nb=16):
    """Synthetic Part 1 reg dir + M3 exact-phi (phix) files + Gram and Adam json under the mock root ('adam' world)."""
    from safetensors.numpy import save_file
    rows = synth_p1_rows(rng, "adam", Gs)
    reg = Path(root) / "runs/firstgrad/part1_reg_i42"
    m3 = Path(root) / "runs/firstgrad/m3"
    reg.mkdir(parents=True, exist_ok=True)
    m3.mkdir(parents=True, exist_ok=True)
    inv = {v: k for k, v in M3_STARTS.items()}
    for s in "BC":
        (reg / f"{s}.jsonl").write_text("".join(json.dumps({**r, "set": s}) + "\n" for r in rows[s]))
    to_phix = {"raw": "v0_lora_g{}", "adamv": "adamv_g{}"}
    phis = defaultdict(list)
    with open(reg / "A.jsonl", "w") as fa:
        for r in rows["A"]:
            dn = r["direction_name"]
            if dn in ("v25", "v150"):
                fa.write(json.dumps({**r, "set": "A"}) + "\n")
            elif dn == "raw_g16":
                fa.write(json.dumps({**r, "direction_name": "v0", "set": "A"}) + "\n")
            elif dn == "craw_1":
                fa.write(json.dumps({**r, "direction_name": "rand_1", "set": "A"}) + "\n")
            f, G = fam_of(dn), g_of(dn)
            if f in to_phix:
                base = to_phix[f].format(G)
            elif dn.startswith("craw_"):
                j = int(dn.split("_")[1])
                base = "p1_rand_lora" if j == 1 else f"rand_lora_r{j}"
            elif dn.startswith("cadamv_"):
                base = f"rand_adamv_r{dn.split('_')[1]}"
            else:
                continue
            phis[r["start"]].append({**r, "direction_name": f"{base}_{inv[r['start']]}", "method": "exact"})
    for st, rr in phis.items():
        (m3 / f"phix_{inv[st]}_A.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rr))
        D = 500
        mu = rng.normal(size=D) * 0.05
        Gb = mu + rng.normal(size=(nb, D)) / math.sqrt(D)
        refs = {"v25": mu + rng.normal(size=D) * 0.2, "v150": mu + rng.normal(size=D) * 0.3}
        dd = m3 / "directions" / inv[st]
        dd.mkdir(parents=True, exist_ok=True)
        save_file({"gram_W": Gb @ Gb.T, "gram_B": 2 * Gb @ Gb.T,
                   "ip_W": np.stack([Gb @ refs[k] for k in sorted(refs)], 1),
                   "raw_sq": (Gb ** 2).sum(1), "raw_cross": np.array([Gb[:b].sum(0) @ Gb[b] for b in range(nb)])},
                  str(dd / "m3_gram.safetensors"))
        rnorm = {k: float(np.linalg.norm(v)) for k, v in refs.items()}
        (dd / "m3_gram.json").write_text(json.dumps({"refs": sorted(refs), "ref_norms": rnorm, "n_batches": nb,
                                                     "raw_stats": True, "pool_size": 750,
                                                     "batches": [{"mean_reward": 0.45, "frac_zero_adv_groups": 0.3}] * nb}))
        dirs = {f"{f}_g{G}": mu * (1 + 0.1 * G ** 0.5) + rng.normal(size=D) / math.sqrt(G * D) for f in ("v0_lora", "adamv") for G in Gs}
        (dd / "m3_adam.json").write_text(json.dumps({"refs": sorted(refs), "clip_coef": [1.0] * nb, "directions": {
            k: {"norm": float(np.linalg.norm(v)), "ip": {r: float(v @ refs[r]) for r in refs}} for k, v in dirs.items()}}))
    P(f"  mock M3 files written to {reg} and {m3}")


def dry_mock(args):
    rng = np.random.default_rng(args.seed)
    root = Path(args.root)
    assert (root / "runs/probes/prereg16").exists(), "build the mock root first: python -I prereg/scripts/p15/p15_mock.py <root>"
    write_mock_m3(rng, root, (1, 4, 16, 64))
    score_m3(root / "runs/firstgrad/part1_reg_i42", root / "runs/firstgrad/m3", args.B)
    export_p1(root / "runs/firstgrad/part1_reg_i42", root / "runs/firstgrad/m3", root / "runs/firstgrad/m3_reg")
    gdir = root / "runs/galign"
    if not (gdir / "untrained_i42").exists():
        write_mock_galign(rng, root, gdir)
    score_m4(root, gdir, args.B, dims=(256, 512, 1024), r_list=(4, 12, 0), d0=1024, r0=12)


# =========================================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["m3", "m4", "all"], default="all")
    ap.add_argument("--schema_only", action="store_true")
    ap.add_argument("--dry_run", choices=["synthetic", "mock"])
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--p1_reg", default="runs/firstgrad/part1_reg_i42")
    ap.add_argument("--m3_dir", default="runs/firstgrad/m3")
    ap.add_argument("--galign_dir", default="runs/galign")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--dry_B", type=int, default=200)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--sections", default="1234")
    ap.add_argument("--export_p1", default=None, help="write per-G frozen Part 1 input dirs <base>_g<G> and exit")
    args = ap.parse_args()
    P(f"# score_prereg19_grad.py args={vars(args)}\n# REG19={REG19}")
    t0 = time.time()
    check_frozen()
    if args.export_p1:
        export_p1(Path(args.root) / args.p1_reg, Path(args.root) / args.m3_dir, args.export_p1)
    elif args.dry_run == "synthetic":
        dry_synthetic(args)
    elif args.dry_run == "mock":
        dry_mock(args)
    elif args.schema_only:
        schema(args.root, args.p1_reg, args.m3_dir, args.galign_dir, REG19["TEMP"])
    else:
        if not args.unblind:
            sys.exit("pass --schema_only, --dry_run, or --unblind (after prereg 19 is registered)")
        blind_gate()
        root = Path(args.root)
        if args.part in ("m3", "all"):
            score_m3(root / args.p1_reg, root / args.m3_dir, args.B)
        if args.part in ("m4", "all"):
            score_m4(root, root / args.galign_dir, args.B)
    P(f"\n# done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
