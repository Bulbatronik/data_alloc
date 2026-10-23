#!/usr/bin/env python3
"""prereg 17A, T_unc arm: the SFT items are pool prompts the untrained model never solves and the teacher does.
Reads (CPU only, run via Slurm after the untrained pool probe of submit_prereg17a.sh):
  runs/gsm8k/Llama-3.2-1B-Instruct/traces/p17_self_k8.jsonl   untrained model, 1,000-prompt pool, k = 8, T 0.8
  runs/gsm8k/teachers/qwen7b.jsonl                            teacher traces on the same pool (k = 4; the trace
                                                              filter of the existing teacher arm: stored `correct`)
Rule (registered): base-unsolved = no sample correct under the LENIENT grader (format-insensitive: a marked-only
criterion would count format failures as unsolved); eligible = base-unsolved AND >= 1 correct teacher trace. If more
than 250 are eligible, take 250 by random.Random(17); else all (n is reported). The RL pool of this arm is the rest of
the 1,000-prompt pool (listed_sft). Writes runs/gsm8k/Llama-3.2-1B-Instruct/traces/p17_tunc_ids.json."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p15lib as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TR = ROOT / "runs/gsm8k/Llama-3.2-1B-Instruct/traces"
solved, n = defaultdict(int), defaultdict(int)
for r in L.read_jsonl(TR / "p17_self_k8.jsonl"):
    e = int(r["example_id"])
    n[e] += 1
    solved[e] += L.answer_is_correct(r["completion"], r["gold"], False, "gsm8k")
assert len(n) == 1000 and all(v == 8 for v in n.values()), "expected 1,000 prompts x 8 samples"
teacher = {int(r["example_id"]) for r in L.read_jsonl(ROOT / "runs/gsm8k/teachers/qwen7b.jsonl") if r["correct"]}
unsolved = sorted(e for e in n if solved[e] == 0)
eligible = [e for e in unsolved if e in teacher]
random.Random(17).shuffle(eligible)
ids = sorted(eligible[:250])
(TR / "p17_tunc_ids.json").write_text(json.dumps(ids))
print(f"base-unsolved (lenient pass@8 = 0): {len(unsolved)} of 1000; with a correct teacher trace: {len(eligible)}; "
      f"T_unc n = {len(ids)} (target 250) -> {TR / 'p17_tunc_ids.json'}")
