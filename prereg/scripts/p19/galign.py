#!/usr/bin/env python3
"""
galign.py -- M4 (prereg 19, MECH_PLAN.md) per-item gradient alignment tooling. GPU; standalone; it imports
firstgrad.py (as F, for the starts and the fp32/bf16 model builders) and route_sft_grpo.py (as R) and changes neither.

What it measures (the definitions are fixed here; the statistics live in score_prereg19_grad.py):
  u      the trainable LoRA parameters of the merged-fresh RL parameterisation: B_l (out_l x r) of the fresh LoRA whose
         A_l is drawn by R.fresh_lora after torch.manual_seed(--lora_init_seed) (firstgrad.fresh_lora_factors), on the
         start's weights (base, or base + bf16-merged gold SFT). At B = 0, dL/dA = 0, so the LoRA gradient is all in B:
         dlog pi/dB_l = s * sum_t delta_t (A_l x_t)^T with delta_t = dlog pi/dy_t at module l's output (s = alpha/r = 2).
         D = sum_l out_l * r = 6,029,312 coordinates (112 modules; layout in meta.json).
  g_i    per-sample gradient of log pi_T(y_i | x) (full sequence incl. the stop token, logits / T), evaluated on the fp32
         start model; computed with forward/backward hooks (no dense weight gradients).
  sample K completions per item from the start policy at T (default 0.6 = the temperature at which prereg 14/16 measured
         Delta D), top-p 0.95, top-k off, <= 512 new tokens (MATH 1,024, as prereg 16), stop = R.stop_token_ids, bf16
         sampler built as firstgrad/probe build it; graded lenient and marked with R.answer_is_correct(dataset).
  The per-item success Jacobian is NOT formed here: from the stored per-sample quantities the scorer forms, for any
  grader and any sample split, g_SF(x) = mean_{correct} g_i - mean_{incorrect} g_i = grad logit p_T(x) (score-function
  identity) and its half-sample versions.
Stored per item (shards out_dir/<dtag>/<set>_T<temp>/grad_<c>.safetensors):
  gram   [K, K] fp64  exact <g_i, g_j> in the D-dim B space (so every aggregate's exact norm is c^T gram c)
  sketch [K, d] fp32  S g_i with an OSNAP sparse sign sketch: d = --sketch_dim (8192), nnz = 4 nonzeros per coordinate
                      (one per block of d/nnz rows, sign +-1/sqrt(nnz)); hashes from numpy PCG64(--sketch_seed), so the
                      sketch is the same on every machine and GPU. Smaller d (4096, 2048) are obtained by FOLDING (summing
                      row u and u + d'/nnz within each block), which is again an OSNAP sketch: d-sensitivity is free.
  logp   [K] fp64     log pi_T(y_i | x) (sanity)
Sets (example_ids = the pipeline's, as probe.py / prereg 14 / prereg 16 use them):
  gsm8k (test, 1,319), gsm_plus (testmini numeric, 2,093), svamp (train+test, 1,000), math600 (MATH L1-3 numeric test,
  the 600 items probe.py --n_prompts 600 selects), pool (GSM8K train, the 1,000-prompt RL pool; data_seed 1234).

Usage (from the repo root; GPU job; resumable per shard):
  python runs/theory/scratch/galign.py --stage all --start gold --seed 42 --sets pool gsm8k gsm_plus svamp math600
  python runs/theory/scratch/galign.py --stage all --start untrained --seed 42 --lora_init_seed 42 --sets ...
  --exact_check (smoke only): keeps the exact g_SF vectors of every processed item in memory and compares the
  sketch-based, debiased alignment with the exact one (projection onto the span of the processed pool items).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import firstgrad as F  # noqa: E402
import route_sft_grpo as R  # noqa: E402

TOP_P = 0.95
SET_SPECS = {  # name: (dataset for loading and grading, split, n_prompts selection, max new tokens)
    "gsm8k": ("gsm8k", "test", None, 512),
    "gsm_plus": ("gsm_plus", "test", None, 512),
    "svamp": ("svamp", "test", None, 512),
    "math600": ("math", "test", 600, 1024),
    "pool": ("gsm8k", "train", None, 512),
}


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def set_items(tokenizer, name: str) -> List[dict]:
    """Items of a set in example_id order: {example_id, prompt, gold, dataset, max_new}."""
    dataset, split, n_sel, max_new = SET_SPECS[name]
    train, test = R.prepare_data(tokenizer, dataset, 1000, -1, 1234, "train")
    ds = train if split == "train" else test
    if n_sel and n_sel < len(ds):  # probe.py select_prompts: positions, example_id kept
        ds = ds.select(sorted(random.Random(0).sample(range(len(ds)), n_sel)))
    items = [{"example_id": int(ex["example_id"]), "prompt": ex["prompt"], "gold": ex["gold_answer"],
              "dataset": dataset, "max_new": max_new} for ex in ds]
    return sorted(items, key=lambda it: it["example_id"])


# ---------------------------------------------------------------------------------------------------------- sketch
class Sketch:
    """OSNAP sparse sign sketch S (d x D): coordinate j maps to row k*(d/nnz) + u_k(j) with sign s_k(j)/sqrt(nnz),
    k = 0..nnz-1. E||Sx||^2 = ||x||^2; S is fixed by (seed, D, d, nnz)."""

    def __init__(self, D: int, d: int = 8192, nnz: int = 4, seed: int = 19, device="cuda"):
        assert d % nnz == 0
        self.D, self.d, self.nnz, self.seed = D, d, nnz, seed
        rng = np.random.Generator(np.random.PCG64(seed))
        b = d // nnz
        idx, sgn, h = [], [], hashlib.sha256()
        for k in range(nnz):
            u = rng.integers(0, b, D).astype(np.int64)
            s = (rng.integers(0, 2, D) * 2 - 1).astype(np.float32) / math.sqrt(nnz)
            h.update(u[:4096].tobytes())
            h.update(s[:4096].tobytes())
            idx.append(torch.from_numpy(k * b + u).to(device))
            sgn.append(torch.from_numpy(s).to(device))
        self.idx, self.sgn, self.digest = idx, sgn, h.hexdigest()[:16]

    def apply(self, X: torch.Tensor, rows_per_step: int = 16) -> torch.Tensor:
        Y = torch.zeros(X.shape[0], self.d, device=X.device, dtype=torch.float32)
        for a in range(0, X.shape[0], rows_per_step):
            for k in range(self.nnz):
                Y[a:a + rows_per_step].index_add_(1, self.idx[k], X[a:a + rows_per_step] * self.sgn[k])
        return Y

    def spec(self) -> dict:
        return {"kind": "osnap", "D": self.D, "d": self.d, "nnz": self.nnz, "seed": self.seed, "digest": self.digest,
                "fold": "row k*(d/nnz)+u -> k*(d'/nnz) + (u mod d'/nnz)"}


def fold(Y: np.ndarray, d: int, d_new: int, nnz: int) -> np.ndarray:
    """OSNAP sketch with d rows -> the nested OSNAP sketch with d_new rows (d_new divides d)."""
    m = d // d_new
    b_new = d_new // nnz
    return Y.reshape(*Y.shape[:-1], nnz, m, b_new).sum(-2).reshape(*Y.shape[:-1], d_new)


# --------------------------------------------------------------------------------------- per-sample LoRA-B grads
class GradProbe:
    """Hooks on the 112 LoRA-targeted linears of an fp32 model with frozen weights. In a backward pass of
    sum_b log pi(y_b | x_b) over a micro-batch (one sequence per row; rows do not interact), the hook on module l writes
    s * delta_b^T (x_b A_l^T) (out x r) of every row b into self.X[self.rows[b], offset_l : offset_l + out*r]."""

    def __init__(self, model, factors, scaling: float):
        self.names = sorted(factors)
        self.s = scaling
        self.A = {n: factors[n][0].float().cuda() for n in self.names}
        self.layout, off = [], 0
        for n in self.names:
            out_f, r = factors[n][1].shape
            self.layout.append({"name": n, "out": int(out_f), "r": int(r), "offset": off})
            off += out_f * r
        self.D = off
        self.off = {d["name"]: (d["offset"], d["out"], d["r"]) for d in self.layout}
        self.X, self.rows = None, None
        self.handles = [model.get_submodule(n).register_forward_hook(self._fwd(n)) for n in self.names]
        emb = model.model.embed_tokens
        self.handles.append(emb.register_forward_hook(
            lambda mod, inp, out: out.detach().requires_grad_(True) if torch.is_grad_enabled() else None))

    def _fwd(self, name):
        def hook(mod, inp, out):
            if self.X is None or not torch.is_grad_enabled() or not out.requires_grad:
                return None
            z = inp[0].detach() @ self.A[name].T  # [B, T, r]
            o0, nout, r = self.off[name]

            def bhook(g):
                gb = torch.bmm(g.transpose(1, 2).float(), z)  # [B, out, r] = sum_t delta_t z_t^T per row
                self.X[self.rows, o0:o0 + nout * r] = (self.s * gb).reshape(gb.shape[0], -1)
                return None
            out.register_hook(bhook)
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


# ------------------------------------------------------------------------------------------------------- sampling
@torch.inference_mode()
def sample_ids(model, tokenizer, prompts: Sequence[str], k: int, seed: int, temperature: float, max_new: int):
    """firstgrad.sample_ids with the temperature and length as arguments: k completions per prompt (prompt-major),
    cut after the first stop token."""
    torch.manual_seed(seed)
    stop = R.stop_token_ids(tokenizer)
    enc = R.encode_prompts(tokenizer, list(prompts)).to("cuda")
    out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=TOP_P, top_k=0, num_return_sequences=k,
                         max_new_tokens=max_new, pad_token_id=tokenizer.pad_token_id, eos_token_id=stop)
    rows = out[:, enc["input_ids"].shape[1]:].tolist()
    stops = set(stop)
    res = []
    for row in rows:
        cut = next((i for i, t in enumerate(row) if t in stops), None)
        res.append(row[: cut + 1] if cut is not None else row)
    return res


def set_dir(args, st, name) -> Path:
    return Path(args.out_dir) / st.dtag / f"{name}_T{args.temperature:g}"


def stage_sample(args, st, tokenizer, name, items) -> dict:
    d = set_dir(args, st, name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "samples.jsonl"
    rows = F.read_jsonl(path)
    have = {}
    for r in rows:
        have[r["item"]] = have.get(r["item"], 0) + 1
    done = {e for e, c in have.items() if c == args.k}
    rows = [r for r in rows if r["item"] in done]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    todo = [it for it in items if it["example_id"] not in done]
    if not todo:
        print(f"[sample] {name}: complete ({len(items)} items)", flush=True)
        return {}
    stop = set(R.stop_token_ids(tokenizer))
    model = F.load_sampler(st)
    t0, n = time.time(), 0
    ppb = args.prompts_per_batch if SET_SPECS[name][3] <= 512 else max(1, args.prompts_per_batch // 2)
    for b in range(0, len(todo), ppb):
        chunk = todo[b: b + ppb]
        seed = seed_of(args.sample_seed, "galign", st.tag, name, args.temperature, chunk[0]["example_id"], len(chunk))
        ids = sample_ids(model, tokenizer, [it["prompt"] for it in chunk], args.k, seed, args.temperature,
                         chunk[0]["max_new"])
        out = []
        for j, tok in enumerate(ids):
            it = chunk[j // args.k]
            text = tokenizer.decode(tok, skip_special_tokens=True)
            out.append({"item": it["example_id"], "sample": j % args.k, "set": name, "start": st.dtag,
                        "temperature": args.temperature, "completion": text,
                        "correct_lenient": R.answer_is_correct(text, it["gold"], False, it["dataset"]),
                        "correct_marked": R.answer_is_correct(text, it["gold"], True, it["dataset"]),
                        "stopped": bool(tok) and tok[-1] in stop, "n_tokens": len(tok), "gold": it["gold"],
                        "gen_seed": seed, "token_ids": tok})
        with path.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in out)
        n += len(out)
        print(f"[sample] {name}: {n} completions, {time.time() - t0:.0f}s", flush=True)
    sec = time.time() - t0
    del model
    torch.cuda.empty_cache()
    return {"completions": n, "seconds": sec, "sec_per_1k": 1000 * sec / max(n, 1)}


def stage_grad(args, st, tokenizer, name, items, model, probe, sketch, exact_store) -> dict:
    d = set_dir(args, st, name)
    rows = F.read_jsonl(d / "samples.jsonl")
    by_item: Dict[int, list] = {}
    for r in rows:
        by_item.setdefault(r["item"], []).append(r)
    ids = [it["example_id"] for it in items if len(by_item.get(it["example_id"], [])) == args.k]
    if len(ids) < len(items):
        print(f"[grad] {name}: {len(items) - len(ids)} items without {args.k} samples are skipped", flush=True)
    prompt_ids = {it["example_id"]: tokenizer(it["prompt"])["input_ids"] for it in items}
    t0, n_seq, n_tok = time.time(), 0, 0
    for c0 in range(0, len(ids), args.items_per_chunk):
        shard = d / f"grad_{c0 // args.items_per_chunk:05d}.safetensors"
        chunk = ids[c0: c0 + args.items_per_chunk]
        if shard.exists() and not (exact_store is not None):
            continue
        samp = [r for e in chunk for r in sorted(by_item[e], key=lambda r: r["sample"])]
        seqs = [(prompt_ids[r["item"]], r["token_ids"]) for r in samp]
        n = len(seqs)
        probe.X = torch.zeros(n, probe.D, device="cuda", dtype=torch.float32)
        logp = torch.zeros(n, dtype=torch.float64)
        for idx in F.make_batches([len(a) + len(b) for a, b in seqs], args.token_budget):
            probe.rows = torch.tensor(idx, device="cuda")
            with torch.enable_grad():
                lp = F.seq_logps(model, seqs, idx, args.temperature, chunk=1024)
                lp.sum().backward()
            logp[idx] = lp.detach().cpu()
        X = probe.X
        probe.X = None
        K = args.k
        if args.verify_grad > 0 and c0 == 0:  # hooks vs firstgrad.merged_gradient (dense dL/dW) mapped by s A^T
            errs = []
            for i in range(min(args.verify_grad, n)):
                Gd = F.merged_gradient(model, probe.names, seqs, [1.0 if j == i else 0.0 for j in range(n)],
                                       temperature=args.temperature, token_budget=args.token_budget)
                ref = torch.cat([(probe.s * Gd[nm] @ probe.A[nm].T).reshape(-1) for nm in probe.names])
                errs.append(float((X[i] - ref).norm() / ref.norm()))
                del Gd
            print(f"[verify_grad] {name}: relative error of hook gradients vs dense merged_gradient @ s A^T: "
                  f"{['%.2e' % e for e in errs]}", flush=True)
        gram = torch.stack([(X[i * K:(i + 1) * K].double() @ X[i * K:(i + 1) * K].double().T).cpu()
                            for i in range(len(chunk))])
        sk = sketch.apply(X).view(len(chunk), K, sketch.d).cpu()
        if exact_store is not None:  # smoke only: exact g_SF (lenient) and its halves, for the sketch check
            for i, e in enumerate(chunk):
                y = torch.tensor([bool(r["correct_lenient"]) for r in samp[i * K:(i + 1) * K]], device="cuda")
                Xi = X[i * K:(i + 1) * K]
                if 0 < int(y.sum()) < K:
                    g = Xi[y].mean(0) - Xi[~y].mean(0)
                    exact_store.setdefault(name, {})[e] = (g.cpu(), sk[i], gram[i], y.cpu())
        save_file({"items": torch.tensor(chunk, dtype=torch.int64), "gram": gram.contiguous(),
                   "sketch": sk.contiguous(), "logp": logp.view(len(chunk), K).contiguous()}, str(shard))
        del X
        n_seq += n
        n_tok += sum(len(a) + len(b) for a, b in seqs)
        if (c0 // args.items_per_chunk) % 10 == 0:
            print(f"[grad] {name}: {c0 + len(chunk)}/{len(ids)} items, {time.time() - t0:.0f}s, "
                  f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    sec = time.time() - t0
    return {"sequences": n_seq, "tokens": n_tok, "seconds": sec, "sec_per_1k_seq": 1000 * sec / max(n_seq, 1),
            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}


def exact_check(exact_store, sketch, nnz, rs=(2, 4, 8, 0)):
    """Smoke-only sanity: alignment of each non-pool item's exact g_SF with the span (or top-r subspace) of the pool
    items' exact g_SF, against the sketch estimate debiased for the sketch's k/d inflation (k = subspace dimension).
    Also: sketched vs exact per-sample Gram entries."""
    pool = exact_store.get("pool", {})
    if len(pool) < 3:
        print("[exact_check] fewer than 3 pool items with both classes: skipped", flush=True)
        return {}

    def sf_coef(y):
        y = y.numpy().astype(bool)
        return np.where(y, 1.0 / y.sum(), -1.0 / (~y).sum())            # g_SF = sum_i c_i g_i

    Gp = torch.stack([v[0] for v in pool.values()]).double()            # [m, D] exact pool g_SF
    _, _, Vt = torch.linalg.svd(Gp, full_matrices=False)
    dds = (sketch.d, sketch.d // 2, sketch.d // 4)
    Wt = {}
    for dd in dds:  # top right-singular vectors of the sketched pool matrix
        Pk = np.stack([sf_coef(v[3]) @ fold(v[1].numpy().astype(np.float64), sketch.d, dd, nnz) for v in pool.values()])
        Wt[dd] = np.linalg.svd(Pk, full_matrices=False)[2]
    out = {"n_pool": len(pool), "rows": []}
    for name, store in exact_store.items():
        if name == "pool":
            continue
        for e, (g, sk_i, gram_i, y) in store.items():
            c = sf_coef(y)
            g = g.double()
            gn2 = float(c @ gram_i.numpy() @ c)                           # exact |g|^2 from the Gram
            row = {"set": name, "item": e, "norm2_gram_vs_vector": gn2 / float(g @ g)}
            for r in rs:
                k = len(pool) if r == 0 else min(r, len(pool))
                proj = Vt[:k] @ g
                exact = float(proj @ proj) / float(g @ g)
                for dd in dds:
                    pr = Wt[dd][:k] @ (c @ fold(sk_i.numpy().astype(np.float64), sketch.d, dd, nnz))
                    raw = float(pr @ pr) / gn2
                    deb = (raw - k / dd) / (1 - k / dd)
                    row[f"r{r or 'full'}_d{dd}"] = {"exact": exact, "sketch_raw": raw, "sketch_debiased": deb}
            out["rows"].append(row)
    for key in sorted({k for row in out["rows"] for k in row if k.startswith("r")}):
        ex = np.array([row[key]["exact"] for row in out["rows"]])
        db = np.array([row[key]["sketch_debiased"] for row in out["rows"]])
        rw = np.array([row[key]["sketch_raw"] for row in out["rows"]])
        print(f"[exact_check] {key}: n {len(ex)}  mean a^2 exact {ex.mean():.4f}  sketch raw {rw.mean():.4f}  "
              f"debiased {db.mean():.4f}  max |debiased - exact| {np.abs(db - ex).max():.4f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["sample", "grad", "all"], required=True)
    p.add_argument("--start", choices=["untrained", "gold"], required=True)
    p.add_argument("--seed", type=int, default=42, help="gold: SFT adapter seed (gold-250).")
    p.add_argument("--lora_init_seed", type=int, default=None, help="fresh-LoRA init seed (default --seed).")
    p.add_argument("--run_dir", default=None, help="optional: the merged-fresh RL run (checked against the start).")
    p.add_argument("--sets", nargs="+", default=["pool", "gsm8k", "gsm_plus", "svamp", "math600"],
                   choices=sorted(SET_SPECS))
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n_items", type=int, default=-1, help="first n items of each set (smoke); -1 = all")
    p.add_argument("--prompts_per_batch", type=int, default=32, help="sampling (halved for MATH's 1,024 tokens)")
    p.add_argument("--sample_seed", type=int, default=19)
    p.add_argument("--token_budget", type=int, default=4096, help="grad: tokens per backward micro-batch")
    p.add_argument("--items_per_chunk", type=int, default=4)
    p.add_argument("--sketch_dim", type=int, default=8192)
    p.add_argument("--sketch_nnz", type=int, default=4)
    p.add_argument("--sketch_seed", type=int, default=19)
    p.add_argument("--no_tf32", action="store_true", help="grad: TF32 off (default on: no finite differences here)")
    p.add_argument("--exact_check", action="store_true", help="smoke only: exact vs sketched alignment")
    p.add_argument("--verify_grad", type=int, default=0, help="smoke: check this many per-sample gradients "
                   "against firstgrad.merged_gradient (dense dW, mapped by s A^T)")
    p.add_argument("--out_dir", default="runs/galign")
    p.add_argument("--timings_out", default=None)
    args = p.parse_args()

    st = F.make_start(args.start, args.seed, "merged", args.lora_init_seed, None, args.run_dir)
    if st.run_dir:
        F.check_run(st)
    tokenizer = R.load_tokenizer(F.model_args())
    timings = {"start": st.dtag, "temperature": args.temperature, "k": args.k, "n_items": args.n_items,
               "gpu": torch.cuda.get_device_name(0), "sample": {}, "grad": {}}
    items = {}
    for name in args.sets:
        items[name] = set_items(tokenizer, name)
        if args.n_items > 0:
            items[name] = items[name][: args.n_items]
    if args.stage in ("sample", "all"):
        for name in args.sets:
            timings["sample"][name] = stage_sample(args, st, tokenizer, name, items[name])
    if args.stage in ("grad", "all"):
        factors, s = F.start_lora(st)
        model = F.load_start_fp32(st, factors, s)           # = start weights (B = 0), fp32, eval, frozen
        if not args.no_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        probe = GradProbe(model, factors, s)
        sketch = Sketch(probe.D, args.sketch_dim, args.sketch_nnz, args.sketch_seed)
        exact_store = {} if args.exact_check else None
        for name in args.sets:
            meta = {"start": st.dtag, "kind": st.kind, "seed": st.seed, "init_seed": st.init_seed, "sft_adapter": st.sft,
                    "set": name, "dataset": SET_SPECS[name][0], "temperature": args.temperature, "top_p": TOP_P,
                    "k": args.k, "max_new": SET_SPECS[name][3], "scaling": s, "D": probe.D, "layout": probe.layout,
                    "sketch": sketch.spec(), "tf32": not args.no_tf32, "token_budget": args.token_budget,
                    "items_per_chunk": args.items_per_chunk, "sample_seed": args.sample_seed,
                    "objective": "per-sample d log pi_T(y|x) / dB, full sequence incl. stop token, logits / T"}
            set_dir(args, st, name).mkdir(parents=True, exist_ok=True)
            (set_dir(args, st, name) / "meta.json").write_text(json.dumps(meta, indent=1))
            timings["grad"][name] = stage_grad(args, st, tokenizer, name, items[name], model, probe, sketch, exact_store)
        if exact_store is not None:
            res = exact_check(exact_store, sketch, args.sketch_nnz)
            (Path(args.out_dir) / st.dtag / "exact_check.json").write_text(json.dumps(res, indent=1, default=float))
        probe.remove()
    print(json.dumps(timings, default=str), flush=True)
    if args.timings_out:
        with open(args.timings_out, "a") as f:
            f.write(json.dumps(timings, default=str) + "\n")


if __name__ == "__main__":
    main()
