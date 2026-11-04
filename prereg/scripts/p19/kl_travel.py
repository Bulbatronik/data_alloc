#!/usr/bin/env python3
"""KL travelled by a GRPO checkpoint from its start, per item, by teacher forcing (prereg 19, M5).

For a checkpoint pi_t (a probe.py-loadable adapter) and its start pi_0 (the same model with the adapter disabled:
for a --merge_sft_start checkpoint that is the merged SFT start, which is exactly TRL's KL reference, a B = 0 copy of
the fresh LoRA; for RL alone it is the untrained model; for a plain SFT adapter it is the base), every sample y of x
drawn from pi_t is scored under both models at a scoring temperature T (logits / T for both):

  lr@T = sum_pos [log q_t(y_pos | x, y_<pos) - log q_0(y_pos | x, y_<pos)]      (one-sample Monte Carlo KL)
  kl@T = sum_pos KL(q_t(. | x, y_<pos) || q_0(. | x, y_<pos))                   (Rao-Blackwellised: exact over the
                                                                               vocabulary at each visited prefix)

with q = softmax(logits / T). Both are unbiased for the sequence-level KL(q_t || q_0) when y ~ q_t; kl is >= 0 and has
far lower variance (primary). The per-item KL travelled is the mean over the item's samples.

Temperature (the caveat). The probe samples are drawn at T 0.6 with top-p 0.95. Scored at T = 0.6 (the default first
scoring temperature = the samples' own temperature) the estimate is the KL between the T-0.6-tempered policies, exact
apart from the top-p truncation of the sampler (not modelled: a token outside pi_0's nucleus would make the truncated
KL infinite). Scored at T = 1 on the same samples, kl@1 is the untempered per-token KL summed along the T-0.6 paths: a
non-negative divergence, but not KL(pi_t || pi_0) (that needs T-1 samples), and lr@1 is not a KL at all. TRL's own
penalty is on the T-0.8-tempered policies (GRPO's sampling temperature), averaged per token, on the training pool.
So kl@0.6 is the primary; kl@1 is reported.

Inputs: probe.py rows (adapter, example_id, sample, gold, stopped, completion), or a training job's predictions.jsonl
(--job_adapter: row i = item i // k), or fresh samples (--generate K, which keeps the exact token ids and also scores
the decode -> re-encode round trip that the file modes rely on). Decoded completions lost their stop token: it is
re-appended (tokenizer EOS) when the row says stopped.

Usage (GPU via Slurm; outputs skip adapters already in --out, so a job can be resubmitted):
  python runs/theory/scratch/kl_travel.py --samples runs/probes/prereg19_kl/b0.1_s100.jsonl \
      --out runs/probes/prereg19_kl/kl_b0.1_s100.jsonl
"""
import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import probe as PR  # noqa: E402
import route_sft_grpo as R  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--prompt_style", default="train")
    p.add_argument("--max_train_samples", type=int, default=1000)
    p.add_argument("--data_seed", type=int, default=1234)
    p.add_argument("--samples", nargs="*", default=[], help="probe.py output files (rows of every adapter in them)")
    p.add_argument("--adapters", nargs="*", default=[], help="globs: keep only these adapters (default: all in --samples)")
    p.add_argument("--job_adapter", default=None, help="--samples is ONE training-job predictions.jsonl of this adapter")
    p.add_argument("--job_k", type=int, default=4)
    p.add_argument("--generate", type=int, default=0, help="K > 0: sample K fresh completions per item from each --adapters")
    p.add_argument("--temperature", type=float, default=0.6, help="sampling temperature for --generate")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gen_batch_size", type=int, default=128, help="samples per generate call (--generate)")
    p.add_argument("--score_temperatures", type=float, nargs="*", default=None,
                   help="default: the samples' temperature, then 1.0")
    p.add_argument("--n_items", type=int, default=-1, help="> 0: a fixed random subset of the items (random.Random(--item_seed))")
    p.add_argument("--item_seed", type=int, default=19)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_samples(args):
    """{adapter: [(example_id, sample, gold, stopped, completion, ids or None)]} and the samples' temperature."""
    by, temps = {}, set()
    if args.job_adapter:
        assert len(args.samples) == 1, "--job_adapter takes one predictions.jsonl"
        rows = [json.loads(l) for l in open(args.samples[0]) if l.strip()]
        k = args.job_k
        assert len(rows) % k == 0
        by[args.job_adapter] = [(i // k, r["sample"], r["gold"], bool(r["stopped"]), r["completion"], None)
                                for i, r in enumerate(rows)]
        temps.add(None)
        return by, temps
    keep = {p for g in args.adapters for p in glob.glob(g)} if args.adapters else None
    for f in args.samples:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            a = r.get("adapter")
            if a is None or (keep is not None and a not in keep):
                continue
            temps.add(r.get("temperature"))
            by.setdefault(a, []).append((int(r["example_id"]), int(r["sample"]), r["gold"], bool(r["stopped"]),
                                         r["completion"], None))
    return by, temps


@torch.inference_mode()
def generate_with_ids(model, tokenizer, prompts, k, args):
    """K samples per prompt (probe settings: top-k 0), keeping token ids up to and including the first stop token."""
    stop = set(R.stop_token_ids(tokenizer))
    out = []
    per = max(1, args.gen_batch_size // k)  # prompts per generate call (probe.py: --batch_size 128 // k)
    for s in range(0, len(prompts), per):
        enc = R.encode_prompts(tokenizer, prompts[s:s + per]).to("cuda")
        gen = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0,
                             num_return_sequences=k, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=sorted(stop))
        gen = gen[:, enc["input_ids"].shape[1]:].tolist()
        for j, g in enumerate(gen):
            cut = next((i for i, t in enumerate(g) if t in stop), None)
            ids = g[:cut + 1] if cut is not None else g  # no stop token: ran to max_new_tokens, so no padding
            out.append((s + j // k, j % k, ids, cut is not None))
    return out


def cont_ids(tokenizer, completion, stopped):
    ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    return ids + [tokenizer.eos_token_id] if stopped else ids


@torch.inference_mode()
def score(model, tokenizer, prompts, conts, temps, batch_size):
    """Per continuation: {f'lr@{T}', f'kl@{T}'} with pi_t = model (adapter on), pi_0 = model with the adapter disabled."""
    res = []
    for s in range(0, len(prompts), batch_size):
        P, C = prompts[s:s + batch_size], conts[s:s + batch_size]
        seqs = [tokenizer(p)["input_ids"] + c for p, c in zip(P, C)]  # as R.encode_prompts / R.token_logprobs
        ids = torch.full((len(seqs), max(map(len, seqs))), tokenizer.pad_token_id)
        mask = torch.zeros_like(ids)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q)
            mask[i, :len(q)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        lt = model(input_ids=ids, attention_mask=mask).logits
        with model.disable_adapter():
            l0 = model(input_ids=ids, attention_mask=mask).logits
        for i, (q, c) in enumerate(zip(seqs, C)):
            n, k = len(q), len(c)
            r = {"n_tok": k}
            if k == 0:
                res.append({**r, **{f"{m}@{T:g}": 0.0 for T in temps for m in ("lr", "kl")}})
                continue
            y = torch.tensor(c, device="cuda")
            at, a0 = lt[i, n - k - 1:n - 1].float(), l0[i, n - k - 1:n - 1].float()
            for T in temps:
                gt, g0 = (at / T).log_softmax(-1), (a0 / T).log_softmax(-1)
                pos = torch.arange(k, device="cuda")
                r[f"lr@{T:g}"] = float((gt[pos, y] - g0[pos, y]).sum())
                r[f"kl@{T:g}"] = float((gt.exp() * (gt - g0)).sum())
            res.append(r)
        del lt, l0
    return res


def main():
    args = parse_args()
    tokenizer = R.load_tokenizer(SimpleNamespace(dataset=args.dataset, model=args.model))
    _, test = R.prepare_data(tokenizer, args.dataset, args.max_train_samples, -1, args.data_seed)
    prompt = {int(e): R.render_question(tokenizer, q, args.prompt_style) for e, q in zip(test["example_id"], test["question"])}
    gold = dict(zip((int(e) for e in test["example_id"]), test["gold_answer"]))
    items = sorted(prompt)
    if args.n_items > 0:
        items = sorted(random.Random(args.item_seed).sample(items, args.n_items))
    keep_items = set(items)

    if args.generate:
        by = {a: None for g in args.adapters for a in sorted(glob.glob(g))}
        temps_in = {args.temperature}
    else:
        by, temps_in = load_samples(args)
    t_samp = next(iter(temps_in)) if len(temps_in) == 1 else None
    temps = args.score_temperatures or ([t_samp] if t_samp else []) + [1.0]
    temps = list(dict.fromkeys(float(t) for t in temps))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["adapter"] for l in out.open()} if out.exists() else set()
    print(f"# kl_travel: {len(by)} adapters, {len(items)} items, sample T {sorted(map(str, temps_in))}, scoring T {temps}", flush=True)
    margs = SimpleNamespace(model=args.model, dataset=args.dataset, merged_start=False)
    for a, rows in by.items():
        if a in done:
            continue
        t0 = time.time()
        model = PR.load_model(margs, tokenizer, a)
        info = R.merged_start_info(a)
        start = (info["sft_adapter"] or "untrained") if info else "base (adapter disabled)"
        if args.generate:
            gen = generate_with_ids(model, tokenizer, [prompt[e] for e in items], args.generate, args)
            rows = []
            for j, s_, ids, stopped in gen:
                e = items[j]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                rows.append((e, s_, gold[e], stopped, text, ids))
        rows = [r for r in rows if r[0] in keep_items]
        bad = sum(gold[r[0]] != r[2] for r in rows)
        assert bad == 0, f"{a}: {bad} gold mismatches (item order differs from the samples' run)"
        t1 = time.time()
        conts = [cont_ids(tokenizer, r[4], r[3]) for r in rows]
        sc = score(model, tokenizer, [prompt[r[0]] for r in rows], conts, temps, args.batch_size)
        extra = []
        if args.generate:  # exact sampled ids vs the decode -> re-encode round trip the file modes use
            exact = [r[5] for r in rows]
            sx = score(model, tokenizer, [prompt[r[0]] for r in rows], exact, temps, args.batch_size)
            eos_used = [bool(r[3]) and r[5][-1] == tokenizer.eos_token_id for r in rows]
            extra = [{"exact_" + k: v for k, v in x.items()} | {"retok_equal": c == r[5], "stop_is_eos": u}
                     for x, c, r, u in zip(sx, conts, rows, eos_used)]
        t2 = time.time()
        with out.open("a") as f:
            for j, (r, s_) in enumerate(zip(rows, sc)):
                f.write(json.dumps({"adapter": a, "start": start,
                                    "example_id": r[0], "sample": r[1], "stopped": r[3], **s_,
                                    **(extra[j] if extra else {}), "score_temperatures": temps}) + "\n")
        n_it = len({r[0] for r in rows})
        T0 = f"{temps[0]:g}"
        mean_kl = sum(x[f"kl@{T0}"] for x in sc) / max(len(sc), 1)
        mean_lr = sum(x[f"lr@{T0}"] for x in sc) / max(len(sc), 1)
        print(f"{a}: {len(rows)} samples / {n_it} items; mean kl@{T0} {mean_kl:.4f}, lr@{T0} {mean_lr:.4f} nats/sample; "
              f"load+gen {t1 - t0:.0f}s, score {t2 - t1:.1f}s ({(t2 - t1) / max(len(rows), 1) * 1000:.0f} ms/sample)", flush=True)
        if extra:
            eq = sum(x["retok_equal"] for x in extra) / len(extra)
            ekl = sum(x[f"exact_kl@{T0}"] for x in extra) / len(extra)
            P = lambda k: sum(x[k] for x in extra) / len(extra)
            print(f"   round trip: re-encoded ids == sampled ids for {eq:.3f} of samples; stop token == EOS for "
                  f"{P('stop_is_eos'):.3f}; mean exact kl@{T0} {ekl:.4f} vs re-encoded {mean_kl:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
