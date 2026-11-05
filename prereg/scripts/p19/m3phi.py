#!/usr/bin/env python3
"""m3phi.py -- exact phi for many fresh-LoRA-geometry directions in ONE per-sample gradient pass (prereg 19 M3).

Every M3 direction is a merged-weight change dW_l = U_l A_l with A_l the start's fresh-LoRA A (v0_lora_g<G>,
sign_g<G>, adamv_g<G>, adamv25_g<G>, adam_run1/2, rand_lora*, rand_sign_r*, rand_adamv*_r*; firstgrad.py
--n_batches --adam_dirs). For such a direction
    phi_v(x, y) = d/de log pi_T,{W + e dW}(y | x) |_{e=0} = sum_l <dlog pi/dW_l, U_l A_l> = sum_l <dlog pi/dW_l A_l^T, U_l>
                = <g_B(x, y), U> / s,
with g_B = s * dlog pi/dW A^T the per-sample LoRA-B gradient that galign.GradProbe computes with hooks. So phi along all
K directions costs one forward + backward per sample, instead of 2 forward passes per direction (firstgrad's central
finite difference, which approximates the same derivative; --compare checks the two agree).
Model: the start's fp32 weights (firstgrad.load_start_fp32), TF32 OFF, eval mode; T = 0.8; full sequence incl. the stop
token: the same log pi_T as firstgrad's phi. Samples: the registered Part 1 set A (--samples_dir).
Output: <out_dir>/phix_<dtag>_A.jsonl, firstgrad's phi row format plus "method": "exact"; resumable per sample.
Directions whose V is not exactly A (v<T>, sft) are skipped: their FD phi exists in Part 1.

  python runs/theory/scratch/m3phi.py --start gold --seed 42 --dirs_dir runs/firstgrad/m3 \
      --extra_dir runs/firstgrad/part1 --samples_dir runs/firstgrad/part1 [--compare runs/firstgrad/part1/phi_<dtag>_A.jsonl]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import firstgrad as F  # noqa: E402
import galign as GA  # noqa: E402
import route_sft_grpo as R  # noqa: E402


def lora_dirs(ddir: Path, A: dict, names, patterns, tag=""):
    """{name: U stacked in the probe layout} for every direction file in ddir whose V == A in every module."""
    out = {}
    for p in sorted(ddir.glob("*.safetensors")):
        name = p.stem
        if not any(fnmatch.fnmatch(name, pat) for pat in patterns) or name.startswith("m3_"):
            continue
        d = F.load_direction(p, device="cpu")
        if sorted(d) != sorted(names) or not all("V" in d[n] and torch.equal(d[n]["V"].float(), A[n].cpu())
                                                  for n in names):
            print(f"[m3phi] {name}: not in the fresh-LoRA geometry (V != A), skipped", flush=True)
            continue
        out[tag + name] = torch.cat([d[n]["U"].float().reshape(-1) for n in names])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", choices=["untrained", "gold"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lora_init_seed", type=int, default=None)
    p.add_argument("--dirs_dir", default="runs/firstgrad/m3", help="M3 out_dir (directions/<dtag>/); output goes here")
    p.add_argument("--extra_dir", default="runs/firstgrad/part1", help="Part 1 out_dir: v0_lora, rand_lora (bridge)")
    p.add_argument("--extra_names", nargs="*", default=["v0_lora", "rand_lora"])
    p.add_argument("--samples_dir", default="runs/firstgrad/part1")
    p.add_argument("--patterns", nargs="+", default=["v0_lora_g*", "sign_g*", "adamv_g*", "adamv25_g*", "adam_run*",
                                                     "rand_lora_r*", "rand_sign_r*", "rand_adamv_r*", "rand_adamv25_r*"])
    p.add_argument("--n_items", type=int, default=-1, help="first n items of set A (smoke)")
    p.add_argument("--token_budget", type=int, default=4096)
    p.add_argument("--compare", nargs="*", default=[], help="FD phi files to compare with (same direction names)")
    p.add_argument("--timings_out", default=None)
    args = p.parse_args()

    st = F.make_start(args.start, args.seed, "merged", args.lora_init_seed)
    tokenizer = R.load_tokenizer(F.model_args())
    _, test = F.load_data(tokenizer)
    samples = F.read_jsonl(Path(args.samples_dir) / f"samples_{st.tag}_A.jsonl")
    if args.n_items > 0:
        keep = sorted({r["item"] for r in samples})[: args.n_items]
        samples = [r for r in samples if r["item"] in set(keep)]
    factors, s = F.start_lora(st)
    names = sorted(factors)
    A = {n: factors[n][0].float() for n in names}
    dirs = lora_dirs(Path(args.dirs_dir) / "directions" / st.dtag, A, names, args.patterns)
    dirs.update(lora_dirs(Path(args.extra_dir) / "directions" / st.dtag, A, names, args.extra_names, tag="p1_"))
    dnames = sorted(dirs)
    print(f"[m3phi] {st.dtag}: {len(samples)} samples, {len(dnames)} directions: {dnames}", flush=True)

    model = F.load_start_fp32(st, factors, s)   # TF32 off (firstgrad.set_precise)
    probe = GA.GradProbe(model, factors, s)
    assert [d["name"] for d in probe.layout] == names
    Umat = torch.stack([dirs[k] for k in dnames]).cuda()           # [K, D]

    out = Path(args.dirs_dir) / f"phix_{st.dtag}_A.jsonl"
    head = Path(args.dirs_dir) / f"phix_{st.dtag}_A.json"
    done = set()
    if out.exists() and head.exists() and json.loads(head.read_text())["directions"] == dnames:
        done = {(r["item"], r["sample"]) for r in F.read_jsonl(out) if r["direction_name"] == f"{dnames[0]}_{st.dtag}"}
    else:
        out.write_text("")
        head.write_text(json.dumps({"start": st.dtag, "directions": dnames, "method": "exact", "temperature": F.TEMPERATURE,
                                    "samples": str(Path(args.samples_dir) / f"samples_{st.tag}_A.jsonl")}, indent=1))
    todo = [r for r in samples if (r["item"], r["sample"]) not in done]
    prompts = {ex["example_id"]: tokenizer(ex["prompt"])["input_ids"] for ex in test}
    seqs = [(prompts[r["item"]], r["token_ids"]) for r in todo]
    t0, n = time.time(), 0
    for idx in F.make_batches([len(a) + len(b) for a, b in seqs], args.token_budget):
        probe.X = torch.zeros(len(idx), probe.D, device="cuda")
        probe.rows = torch.arange(len(idx), device="cuda")
        with torch.enable_grad():
            F.seq_logps(model, seqs, idx, F.TEMPERATURE, chunk=1024).sum().backward()
        phi = (probe.X @ Umat.T / s).double().cpu()                  # [b, K]
        probe.X = None
        rows = []
        for j, i in enumerate(idx):
            r = todo[i]
            for k, dn in enumerate(dnames):
                rows.append({"item": r["item"], "sample": r["sample"], "start": st.tag, "direction_name": f"{dn}_{st.dtag}",
                             "phi": float(phi[j, k]), "correct_lenient": r["correct_lenient"],
                             "correct_marked": r["correct_marked"], "method": "exact"})
        with out.open("a") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        n += len(idx)
        if n % 512 < len(idx):
            print(f"[m3phi] {n}/{len(todo)} samples, {time.time() - t0:.0f}s", flush=True)
    sec = time.time() - t0
    print(f"[m3phi] {st.dtag}: {n} samples x {len(dnames)} directions in {sec:.0f}s "
          f"({1000 * sec / max(n, 1):.1f}s per 1k samples); peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          flush=True)
    # numerical agreement with firstgrad's finite-difference phi (same derivative), where names overlap
    ex = {(r["direction_name"], r["item"], r["sample"]): r["phi"] for r in F.read_jsonl(out)}
    for f in args.compare:
        fd = {}
        for r in F.read_jsonl(f):
            base = r["direction_name"][: -len(st.dtag) - 1]
            for cand in (base, "p1_" + base):
                key = (f"{cand}_{st.dtag}", r["item"], r["sample"])
                if key in ex:
                    fd.setdefault(cand, []).append((r["phi"], ex[key]))
        for nm, pairs in sorted(fd.items()):
            a = torch.tensor(pairs, dtype=torch.float64)
            corr = float(torch.corrcoef(a.T)[0, 1]) if len(a) > 2 else float("nan")
            rel = float((a[:, 0] - a[:, 1]).pow(2).mean().sqrt() / a[:, 1].pow(2).mean().sqrt())
            sc = float((a[:, 0] * a[:, 1]).sum() / (a[:, 1] ** 2).sum())
            print(f"[m3phi] FD vs exact {nm} ({Path(f).name}): n {len(a)}  corr {corr:.6f}  rel RMS diff {rel:.2e}  "
                  f"FD/exact scale {sc:.4f}", flush=True)
    if args.timings_out:
        with open(args.timings_out, "a") as f:
            f.write(json.dumps({"tool": "m3phi", "start": st.dtag, "samples": n, "directions": len(dnames),
                                "seconds": sec, "gpu": torch.cuda.get_device_name(0)}) + "\n")
    probe.remove()


if __name__ == "__main__":
    main()
