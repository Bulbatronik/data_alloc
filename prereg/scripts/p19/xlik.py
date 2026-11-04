#!/usr/bin/env python3
"""M2 (prereg 19 tooling): teacher-forced cross-likelihoods of STORED completions under the other models of an arm.

For one family and seed, and one or more SFT arms, the roles are
    U0 = untrained (no adapter)                     U1 = RL alone at the final step (same seed)
    S0:<arm> = the arm's SFT start (after SFT)      S1:<arm> = the arm's SFT start after RL (final step)
Each role's own stored samples (prereg-14 Part A / prereg-17B probe files: T 0.6, top-p 0.95) are teacher-forced under
every scorer role of the same arm (4 x 4 per arm; --cross adds the pairs between arms of the same seed).

Generation fidelity (as probe.py generated the samples):
  - models are loaded by probe.load_model itself (R.load_fresh_model: neutral generation config; a --merge_sft_start
    GRPO adapter is rebuilt as base -> merged SFT LoRA -> GRPO adapter, exactly as at generation time);
  - prompts are R.render_question(tokenizer, question, prompt_style) on R.prepare_data(..., max_train_samples=1000,
    max_eval_samples=-1, data_seed=1234) test rows, keyed by example_id; the stored gold and prompt_style are asserted;
  - the tokenizer is R.load_tokenizer (Gemma: <end_of_turn> as EOS); teacher forcing follows R.token_logprobs
    (prompt ids from tokenizer(prompt) + completion ids, right padding with an attention mask, positions 0..n-1).
  The completion text is re-tokenised with add_special_tokens=False (the generated ids were not stored); a stopped
  sample gets the tokenizer's EOS appended, whose log-probability is recorded separately (lp*_stop).

Output (jsonl, one row per scorer x source x example_id x sample):
  family, seed, scorer, scorer_adapter, source, source_adapter, example_id, sample, stopped, n_tok (completion tokens,
  without the stop token), lp1 / lp1_stop (sum of log p at T = 1 over completion tokens / of the stop token, null if
  not stopped), lp06 / lp06_stop (the same at T = 0.6, no top-p). No grading is done here (blinding: the scorer
  joins correctness from the probe files with the frozen graders).
Resumable: a (scorer, source) block already complete in --out is skipped.

Usage (GPU via Slurm; run from the repository root):
  python -I runs/theory/scratch/xlik.py --family gsm8k --seed 42 --arms gold_250,teacher_250 --out <jsonl>
  python -I runs/theory/scratch/xlik.py --family llama17b --seed 42 --arms gold,teacher --out <jsonl>
  --n_items 20 (smoke test: the first 20 example_ids), --check_ref 16 (compare with R.token_logprobs on 16 sequences),
  --sanity_only (print only finiteness / range checks and timings, never statistics).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import probe  # noqa: E402  (probe.load_model: the generation path's model loading)
import route_sft_grpo as R  # noqa: E402

G = "runs/gsm8k/Llama-3.2-1B-Instruct"
FAMILIES = {
    "gsm8k": dict(model="unsloth/Llama-3.2-1B-Instruct", dataset="gsm8k", prompt_style="train", step=375, k=4, max_new=512,
                  files={"U0": "runs/probes/prereg14_a/untrained.jsonl", "S0": "runs/probes/prereg14_a/sft.jsonl",
                         "U1": "runs/probes/prereg14_a/s375.jsonl", "S1": "runs/probes/prereg14_a/s375.jsonl"},
                  arms=("gold_50", "gold_250", "gold_500", "teacher_50", "teacher_250", "teacher_500")),
    "llama17b": dict(model="unsloth/Llama-3.2-1B-Instruct", dataset="math35d", prompt_style="boxed_user", step=300, k=8,
                     max_new=1024, fam="llama", arms=("gold", "teacher", "rft")),
    "gemma17b": dict(model="google/gemma-3-1b-it", dataset="math35d", prompt_style="boxed_user", step=300, k=8,
                     max_new=1024, fam="gemma", arms=("gold", "teacher", "rft")),
}
for _f in ("llama17b", "gemma17b"):
    _p = FAMILIES[_f]["fam"]
    FAMILIES[_f]["files"] = {"U0": f"runs/probes/prereg17b/{_p}_mf_untrained.jsonl", "S0": f"runs/probes/prereg17b/{_p}_mf_sft.jsonl",
                             "U1": f"runs/probes/prereg17b/{_p}_mf_s300.jsonl", "S1": f"runs/probes/prereg17b/{_p}_mf_s300.jsonl"}


def adapters(family, arm, seed):
    """{role: adapter path or None} for one arm and seed (paths as stored in the probe files, relative to the repo)."""
    F = FAMILIES[family]
    if family == "gsm8k":
        src, n = arm.split("_")
        q = "-qwen7b" if src == "teacher" else ""
        return {"U0": None, "U1": f"{G}-marked-steps375/seed_{seed}/full_grpo/grpo/checkpoint-375",
                f"S0:{arm}": f"{G}{q}/seed_{seed}/random_sft_{n}/sft",
                f"S1:{arm}": f"{G}-marked-steps375{q}/seed_{seed}/random_sft_{n}/grpo/checkpoint-375"}
    fam, t = F["fam"], F["step"]
    return {"U0": None, "U1": f"runs/math35d/{fam}-p17b-mf-grpo/seed_{seed}/full_grpo/grpo/checkpoint-{t}",
            f"S0:{arm}": f"runs/math35d/{fam}-p17b-mf-{arm}/seed_{seed}/listed_sft/sft",
            f"S1:{arm}": f"runs/math35d/{fam}-p17b-mf-{arm}/seed_{seed}/listed_sft/grpo/checkpoint-{t}"}


def norm(a):
    return None if a is None else os.path.normpath(str(a)).rstrip("/")


def load_samples(family, roles, items, k):
    """{role: [(example_id, sample, completion, stopped, gold)]} from the probe files (one streaming pass per file).
    Only 'adapter', 'example_id', 'sample', 'completion', 'stopped', 'gold', 'prompt_style' are read."""
    F = FAMILIES[family]
    want = defaultdict(list)  # file -> [(role, adapter)]
    for role, a in roles.items():
        want[F["files"][role.split(":")[0]]].append((role, norm(a)))
    out = {r: [] for r in roles}
    for f, rl in want.items():
        by_ad = {a: r for r, a in rl}
        with open(REPO / f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                x = json.loads(line)
                r = by_ad.get(norm(x.get("adapter")), "__none__")
                if r == "__none__":
                    continue
                e, s = int(x["example_id"]), int(x["sample"])
                if s >= k or (items is not None and e not in items):
                    continue
                assert x["prompt_style"] == F["prompt_style"], (f, x["prompt_style"])
                assert not x.get("force_prefix"), "forced-prefix rows are not generation samples"
                out[r].append((e, s, x["completion"], bool(x["stopped"]), x["gold"]))
    for r, rows in out.items():
        assert rows, f"no samples for role {r} ({roles[r]}) in {F['files'][r.split(':')[0]]}"
        rows.sort()
        ids = sorted({e for e, *_ in rows})
        n_exp = len(ids) * k
        assert len(rows) == n_exp and len(set((e, s) for e, s, *_ in rows)) == n_exp, f"{r}: incomplete samples"
    ref = sorted({e for e, *_ in out["U0"]})
    for r, rows in out.items():
        assert sorted({e for e, *_ in rows}) == ref, f"{r}: item set differs from U0's"
    return out


@torch.inference_mode()
def teacher_force(model, tokenizer, prompt_ids, cont_ids, has_stop, max_tokens, temps=(1.0, 0.6)):
    """Per sequence: {T: (sum of log p over completion tokens excluding the stop token, log p of the stop token or None)}.
    Batches are packed by padded token count (sorted by length); right padding, attention mask, as R.token_logprobs."""
    device = next(model.parameters()).device
    n = len(prompt_ids)
    order = sorted(range(n), key=lambda i: len(prompt_ids[i]) + len(cont_ids[i]))
    res = [None] * n
    pad = R.text_tokenizer(tokenizer).pad_token_id
    i = 0
    tok_total = 0
    while i < n:
        j, L = i, 0
        while j < n:
            Lj = max(L, len(prompt_ids[order[j]]) + len(cont_ids[order[j]]))
            if j > i and Lj * (j - i + 1) > max_tokens:
                break
            L, j = Lj, j + 1
        idx = order[i:j]
        seqs = [prompt_ids[t] + cont_ids[t] for t in idx]
        ids = torch.full((len(seqs), L), pad)
        mask = torch.zeros_like(ids)
        for b, s in enumerate(seqs):
            ids[b, : len(s)] = torch.tensor(s)
            mask[b, : len(s)] = 1
        tok_total += int(mask.sum())
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        vals = []  # per sequence, per T: (content sum, stop log p) as 0-d tensors; one host transfer per batch
        for b, t in enumerate(idx):
            m, c = len(seqs[b]), cont_ids[t]
            kc = len(c)
            if kc == 0:
                vals.append([torch.zeros((), device=device)] * (2 * len(temps)))
                continue
            sl = logits[b, m - kc - 1 : m - 1].float()
            tgt = torch.tensor(c, device=device)[:, None]
            row = []
            for T in temps:
                lp = (sl / T).log_softmax(-1).gather(1, tgt).squeeze(1)
                if has_stop[t]:
                    row += [lp[:-1].sum(), lp[-1]]
                else:
                    row += [lp.sum(), torch.zeros((), device=device)]
            vals.append(row)
        host = torch.stack([torch.stack(r) for r in vals]).tolist()
        for b, t in enumerate(idx):
            h = host[b]
            res[t] = {T: (h[2 * q], h[2 * q + 1] if has_stop[t] else None) for q, T in enumerate(temps)}
        del logits
        i = j
    return res, tok_total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arms", required=True, help="comma-separated arms of the family (e.g. gold_250,teacher_250 or gold,teacher)")
    ap.add_argument("--cross", action="store_true", help="also score every pair between the arms' roles (gold <-> teacher)")
    ap.add_argument("--n_items", type=int, default=-1, help="first N example_ids only (smoke test)")
    ap.add_argument("--max_tokens", type=int, default=16384, help="padded tokens per forward batch")
    ap.add_argument("--check_ref", type=int, default=0, help="compare N sequences with R.token_logprobs (implementation check)")
    ap.add_argument("--sanity_only", action="store_true", help="print only finiteness/range checks and timings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(REPO)  # merged_start.json stores repo-relative SFT adapter paths
    F = FAMILIES[args.family]
    arms = args.arms.split(",")
    assert all(a in F["arms"] for a in arms), arms
    roles = {}
    for a in arms:
        roles.update(adapters(args.family, a, args.seed))
    for r, a in roles.items():
        assert a is None or (REPO / a / "adapter_model.safetensors").exists(), f"missing adapter {a}"
    pairs = set()
    for a in arms:
        rr = ["U0", "U1", f"S0:{a}", f"S1:{a}"]
        pairs |= {(m, x) for m in rr for x in rr}
    if args.cross:
        pairs |= {(m, x) for m in roles for x in roles}
    t0 = time.time()
    tok_args = SimpleNamespace(model=F["model"], dataset=F["dataset"])
    tokenizer = R.load_tokenizer(tok_args)
    _, test = R.prepare_data(tokenizer, F["dataset"], 1000, -1, 1234)
    ids_all = sorted(int(e) for e in test["example_id"])
    items = set(ids_all[: args.n_items]) if args.n_items > 0 else None
    samples = load_samples(args.family, roles, items, F["k"])
    by_id = {int(r["example_id"]): r for r in test.select_columns(["example_id", "question", "gold_answer"])} if items is None else \
        {int(r["example_id"]): r for r in test.filter(lambda ex: ex["example_id"] in items).select_columns(["example_id", "question", "gold_answer"])}
    prompts = {e: R.render_question(tokenizer, r["question"], F["prompt_style"]) for e, r in by_id.items()}
    pids = {e: tokenizer(p)["input_ids"] for e, p in prompts.items()}
    eos = R.text_tokenizer(tokenizer).eos_token_id
    enc = {}
    trunc_ok = trunc_n = 0
    for r, rows in samples.items():
        cl, hs = [], []
        for e, s, text, stopped, gold in rows:
            assert str(gold) == str(by_id[e]["gold_answer"]), f"gold mismatch {r} item {e}"
            c = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not stopped:
                trunc_n += 1
                trunc_ok += abs(len(c) - F["max_new"]) <= 3
            cl.append(c + ([eos] if stopped else []))
            hs.append(stopped)
        enc[r] = (cl, hs)
    print(f"# setup {time.time() - t0:.0f}s; roles {roles}; items {len(prompts)}; samples/role {[len(v) for v in samples.values()]}; "
          f"truncated samples re-tokenised to max_new_tokens +-3: {trunc_ok}/{trunc_n}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = defaultdict(int)
    if out.exists():
        for line in out.open():
            if line.strip():
                x = json.loads(line)
                if x["seed"] == args.seed:
                    done[(x["scorer"], x["source"])] += 1
    timing = []
    nonfinite = total = pos = 0
    scorers = sorted({m for m, _ in pairs}, key=lambda m: (m != "U0", m))
    for m in scorers:
        todo = [x for (mm, x) in sorted(pairs) if mm == m and done[(m, x)] < len(samples[x])]
        if not todo:
            continue
        tl = time.time()
        model = probe.load_model(SimpleNamespace(model=F["model"], dataset=F["dataset"]), tokenizer, roles[m])
        t_load = time.time() - tl
        for x in todo:
            rows = samples[x]
            cl, hs = enc[x]
            pl = [pids[e] for e, *_ in rows]
            torch.cuda.synchronize()
            ts = time.time()
            res, ntok = teacher_force(model, tokenizer, pl, cl, hs, args.max_tokens)
            torch.cuda.synchronize()
            dt = time.time() - ts
            timing.append({"scorer": m, "source": x, "n_seq": len(rows), "n_tok": ntok, "sec": dt, "load_sec": t_load})
            t_load = 0.0
            with out.open("a") as f:
                for (e, s, _, stopped, _), c, r in zip(rows, cl, res):
                    n_tok = len(c) - (1 if stopped else 0)
                    rec = {"family": args.family, "seed": args.seed, "scorer": m, "scorer_adapter": roles[m], "source": x,
                           "source_adapter": roles[x], "example_id": e, "sample": s, "stopped": stopped, "n_tok": n_tok,
                           "lp1": r[1.0][0], "lp1_stop": r[1.0][1], "lp06": r[0.6][0], "lp06_stop": r[0.6][1]}
                    for v in (rec["lp1"], rec["lp06"]):
                        total += 1
                        nonfinite += not math.isfinite(v)
                        pos += v > 1e-6
                    f.write(json.dumps(rec) + "\n")
            print(f"  {m:>16s} <- {x:<16s}: {len(rows)} seqs, {ntok} tokens, {dt:.1f}s "
                  f"({1000 * dt / max(1, len(rows)):.1f} s per 1k seqs; {ntok / max(dt, 1e-9):.0f} tok/s)", flush=True)
        if args.check_ref > 0:
            x = todo[0]
            rows = samples[x][: args.check_ref]
            cl, hs = enc[x]
            pl = [pids[e] for e, *_ in rows]
            ref = R.token_logprobs(model, tokenizer, [prompts[e] for e, *_ in rows], cl[: len(rows)], 1.0, 4)
            mine, _ = teacher_force(model, tokenizer, pl, cl[: len(rows)], hs[: len(rows)], args.max_tokens)
            diffs = []
            for r_, mm, h in zip(ref, mine, hs[: len(rows)]):
                tot = mm[1.0][0] + (mm[1.0][1] or 0.0)
                diffs.append(abs(sum(r_) - tot) / max(1, len(r_)))
            print(f"  implementation check vs R.token_logprobs ({m}, {len(rows)} seqs of {x}): max |diff| per token {max(diffs):.2e}", flush=True)
        del model
        torch.cuda.empty_cache()
    tot_seq = sum(t["n_seq"] for t in timing)
    tot_tok = sum(t["n_tok"] for t in timing)
    tot_sec = sum(t["sec"] for t in timing)
    tot_load = sum(t["load_sec"] for t in timing)
    summ = {"family": args.family, "seed": args.seed, "arms": arms, "blocks": len(timing), "n_seq": tot_seq, "n_tok": tot_tok,
            "score_sec": round(tot_sec, 1), "load_sec": round(tot_load, 1),
            "sec_per_1k_seq": round(1000 * tot_sec / max(1, tot_seq), 2), "tok_per_sec": round(tot_tok / max(tot_sec, 1e-9)),
            "mean_tok_per_seq": round(tot_tok / max(1, tot_seq), 1), "gpu": torch.cuda.get_device_name(0),
            "sanity": {"values": total, "nonfinite": nonfinite, "positive_logp": pos}}
    print("# TIMING " + json.dumps(summ), flush=True)
    print(f"# done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
