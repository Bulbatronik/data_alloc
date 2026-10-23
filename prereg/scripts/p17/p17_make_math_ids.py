#!/usr/bin/env python3
"""prereg 17B: the SFT item lists (n = 500 per arm and student family) from the fixed MATH L3-5 SFT pool.
Rule (registered): order the 1,500 SFT-pool ids by random.Random(17); each arm takes the first 500 ids that have a
trace for that arm:
  gold     every SFT-pool item (Hendrycks solution; --sft_traces gold);
  teacher  items with a correct (marked) Llama-3.1-8B-Instruct trace: data/math35/traces/teacher_llama8b.jsonl;
  rft      items with a correct (marked) own sample of that student: data/math35/traces/{llama1b,gemma1b}.jsonl
           (the dataset engineer's files: one row per covered item, matched on `id`).
If an arm has fewer than 500 covered items it takes all of them, and n is reported. Writes
data/math35/p17b_sft_ids_<arm>_<family>.json and prints n and the pairwise overlaps. CPU only (Slurm)."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/math35"
pool = json.loads((D / "sft_pool_ids.json").read_text())
order = list(pool)
random.Random(17).shuffle(order)


def covered(fname):
    p = D / "traces" / fname
    if not p.exists():
        raise SystemExit(f"missing {p}: the dataset engineer's trace files must exist first")
    return {json.loads(line)["id"] for line in open(p) if line.strip()}


src = {"gold": None, "teacher": "teacher_llama8b.jsonl"}
out = {}
for fam, own in (("llama", "llama1b.jsonl"), ("gemma", "gemma1b.jsonl")):
    for arm, f in (("gold", None), ("teacher", "teacher_llama8b.jsonl"), ("rft", own)):
        cov = set(pool) if f is None else covered(f)
        ids = [i for i in order if i in cov][:500]
        out[(arm, fam)] = ids
        (D / f"p17b_sft_ids_{arm}_{fam}.json").write_text(json.dumps(ids))
        print(f"{fam} {arm}: covered {len(cov & set(pool))} of {len(pool)}; n = {len(ids)}")
    for a, b in (("gold", "teacher"), ("gold", "rft"), ("teacher", "rft")):
        print(f"  {fam} overlap {a}/{b}: {len(set(out[(a, fam)]) & set(out[(b, fam)]))}")
