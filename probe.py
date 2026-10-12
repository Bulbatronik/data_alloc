#!/usr/bin/env python3
"""
Eval-only probes of untrained models and saved adapters (no training).

Greedy test-set accuracy of the untrained model under a given prompt:
    python probe.py --model unsloth/Llama-3.2-1B-Instruct --prompt_style llama_math --out runs/probes/t1.jsonl

Sample saved adapters on the GRPO prompts of their own run (as GRPO would):
    python probe.py --model unsloth/Llama-3.2-1B-Instruct --on grpo_bucket --k 8 --temperature 0.8 \
        --adapters 'runs/gsm8k/Llama-3.2-1B-Instruct/seed_*/*_sft_*/sft' --out runs/probes/t2.jsonl

Probability that answers start with a style's opening (teacher-forced, no generation), e.g. Llama's native style:
    python probe.py --model unsloth/Llama-3.2-1B-Instruct --on grpo_bucket --prefix "## Step" --temperature 0.8 \
        --adapters 'runs/gsm8k/Llama-3.2-1B-Instruct/seed_*/*_sft_*/sft' --out runs/probes/prefix.jsonl

Add --entropy to a sampling probe to also record each completion's mean per-token policy entropy, and
--force_prefix "## Step" to sample answers forced to open with a style's opening (that style's quality and length).

Model-agnostic share of the model's own style after foreign-trace SFT (runs/theory/style_diagnostic.md). The library
of decisive own-style openings is built once from the untrained model vs. --foreign_ref (the student SFT'd on all
foreign traces) and reused:
    python probe.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --on grpo_bucket --n_prompts 256 --k 32 --temperature 0.8 \
        --own_style runs/probes/own_style/SmolLM2-r1.json --foreign_ref runs/gsm8k/SmolLM2-1.7B-Instruct-sftscan-r1/seed_42/full_sft/sft \
        --adapters 'runs/gsm8k/SmolLM2-1.7B-Instruct-sftscan-r1/seed_42/random_sft_*/sft' --out runs/probes/own_style/SmolLM2-r1.jsonl

Geometry3K with a VLM: the question with its diagram (--input image), without it (text), or with the diagram replaced by
its Inter-GPS gold literals as text (caption; needs Inter-GPS's data/geometry3k, see diagram_captions):
    python probe.py --model Qwen/Qwen2.5-VL-3B-Instruct --dataset geometry3k --prompt_style boxed --reward marked \
        --input caption --intergps_dir runs/vlm/intergps --out runs/probes/geo3k_caption.jsonl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import re
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import PeftModel
from PIL import Image

import route_sft_grpo as R

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", choices=["gsm8k", "math", "svamp", "gsm_plus", "geometry3k"], default="gsm8k")
    p.add_argument("--input", choices=["image", "text", "caption"], default="image",
                   help="Image datasets: the question with its image, the question alone, or with the image replaced by "
                        "its Inter-GPS diagram literals as text (caption, needs --intergps_dir).")
    p.add_argument("--intergps_dir", default=None,
                   help="Inter-GPS data/geometry3k (github.com/lupantech/InterGPS, after data/unzip_data.sh).")
    p.add_argument("--reward", choices=["lenient", "marked"], default="lenient", help="Grading, as in route_sft_grpo.py.")
    p.add_argument("--adapters", nargs="*", default=[], help="Globs of adapter dirs; none = untrained model.")
    p.add_argument("--on", choices=["test", "grpo_bucket", "pool"], default="test",
                   help="grpo_bucket = the run's GRPO prompts (from <adapter>/../partition.json); pool = all training problems.")
    p.add_argument("--prompt_style", choices=list(R.PROMPT_STYLES), default="train")
    p.add_argument("--n_prompts", type=int, default=-1)
    p.add_argument("--shard", default=None, help="i/k: probe only the i-th of k contiguous parts of the prompts (split long runs).")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy.")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--prefix", default=None,
                   help="Score log p(completion starts with PREFIX) at --temperature (1 if 0) instead of generating.")
    p.add_argument("--force_prefix", default=None,
                   help="Sample completions that start with this text (e.g. a style's opening), to measure that style's quality.")
    p.add_argument("--entropy", action="store_true",
                   help="Also record each completion's mean per-token policy entropy at --temperature (1 if 0).")
    p.add_argument("--own_style", default=None,
                   help="Library of decisive own-style openings (JSON); built from the pool if missing (needs --foreign_ref).")
    p.add_argument("--foreign_ref", default=None, help="Adapter SFT'd on all foreign traces (reference for --own_style).")
    p.add_argument("--own_style_lam", type=float, default=9.2, help="Cut openings where log p0 - log p_foreign >= this (ln 1e4).")
    p.add_argument("--own_style_len", type=int, default=64, help="Tokens per sampled opening.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_train_samples", type=int, default=1000)
    p.add_argument("--data_seed", type=int, default=1234)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_model(args, tokenizer, adapter: str | None):
    model = R.load_fresh_model(
        SimpleNamespace(model=args.model, dataset=args.dataset, bf16=True, fp16=False, gradient_checkpointing=False)
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.cuda().eval()


def select_prompts(args, tokenizer, train, test, adapter: str | None):
    if args.on == "grpo_bucket":
        partition = json.loads((Path(adapter).parent / "partition.json").read_text())
        ids = set(partition["grpo_example_ids"])
        ds = train.filter(lambda ex: ex["example_id"] in ids)
    elif args.on == "pool":
        ds = train
    else:
        ds = test
    if 0 < args.n_prompts < len(ds):
        ds = ds.select(sorted(random.Random(0).sample(range(len(ds)), args.n_prompts)))
    if args.shard:
        index, count = map(int, args.shard.split("/"))
        ds = ds.shard(num_shards=count, index=index, contiguous=True)
    return ds.map(lambda ex: {"prompt": R.render_question(tokenizer, ex["question"], args.prompt_style)})


def generate(args, model, tokenizer, ds):
    """Yield (row, sample index, completion, stopped) for every generated completion."""
    if args.temperature == 0:
        eval_args = SimpleNamespace(eval_batch_size=args.batch_size, eval_max_new_tokens=args.max_new_tokens,
                                    dataset=args.dataset, reward=args.reward)
        preds = R.evaluate(model, tokenizer, ds, eval_args)["predictions"]
        for row, pred in zip(ds, preds):
            yield row, 0, pred["completion"], pred["stopped"]
        return
    groups = R.sample_probe_completions(
        model, tokenizer, list(ds["prompt"]), k=args.k, batch_size=max(1, args.batch_size // args.k),
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        images=list(ds["images"]) if "images" in ds.column_names else None,
    )
    for row, group in zip(ds, groups):
        for i, text in enumerate(group):
            yield row, i, text, len(R.text_tokenizer(tokenizer)(text)["input_ids"]) < args.max_new_tokens - 1


def own_style_library(args, model, tokenizer, ds):
    """
    Decisive own-style openings per prompt. `model` carries the foreign adapter; with it disabled it is the untrained
    model p0. Each of k openings sampled from p0 is cut at its first token where log p0 - log p_foreign >= lam, so the
    cut is >= e^lam times more likely under p0. Returns {example_id: {"prefixes": [token ids], "m0": p0's mass on them}}.
    """
    prompts = list(ds["prompt"])
    with model.disable_adapter():
        groups = R.sample_probe_completions(model, tokenizer, prompts, k=args.k, batch_size=max(1, args.batch_size // args.k),
                                            max_new_tokens=args.own_style_len, temperature=args.temperature, top_p=args.top_p)
    flat = [(ex_id, prompt, tokenizer(text, add_special_tokens=False)["input_ids"][: args.own_style_len])
            for ex_id, prompt, group in zip(ds["example_id"], prompts, groups) for text in group]
    flat = [f for f in flat if f[2]]
    with model.disable_adapter():
        lp0 = R.token_logprobs(model, tokenizer, [f[1] for f in flat], [f[2] for f in flat], args.temperature, args.batch_size)
    lpf = R.token_logprobs(model, tokenizer, [f[1] for f in flat], [f[2] for f in flat], args.temperature, args.batch_size)

    cuts = {ex_id: {} for ex_id in ds["example_id"]}
    for (ex_id, _, ids), a, b in zip(flat, lp0, lpf):
        ratio = 0.0
        for t, (x, y) in enumerate(zip(a, b), 1):
            ratio += x - y
            if ratio >= args.own_style_lam:
                cuts[ex_id][tuple(ids[:t])] = sum(a[:t])
                break
    return {ex_id: {"prefixes": [list(p) for p in c], "m0": sum(math.exp(v) for v in c.values())} for ex_id, c in cuts.items()}


def own_style_mass(model, tokenizer, prompts, prefix_sets, temperature: float, batch_size: int):
    """Teacher-forced probability, per prompt, that the completion opens with one of the prompt's decisive prefixes."""
    flat = [(i, p) for i, prefixes in enumerate(prefix_sets) for p in prefixes]
    logps = R.token_logprobs(model, tokenizer, [prompts[i] for i, _ in flat], [p for _, p in flat], temperature, batch_size)
    mass = [0.0] * len(prompts)
    for (i, _), lp in zip(flat, logps):
        mass[i] += math.exp(sum(lp))
    return mass


@torch.inference_mode()
def completion_entropy(model, tokenizer, prompts, completions, temperature: float, batch_size: int):
    """Mean per-token entropy (nats) of the policy at `temperature` along each completion."""
    out = []
    for start in range(0, len(prompts), batch_size):
        pairs = [(tokenizer(p)["input_ids"], tokenizer(c, add_special_tokens=False)["input_ids"])
                 for p, c in zip(prompts[start : start + batch_size], completions[start : start + batch_size])]
        seqs = [p + c for p, c in pairs]
        ids = torch.full((len(seqs), max(map(len, seqs))), tokenizer.pad_token_id)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq)
            mask[i, : len(seq)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        for i, (p, c) in enumerate(pairs):
            logp = (logits[i, len(p) - 1 : len(p) + len(c) - 1].float() / temperature).log_softmax(-1)
            out.append(-(logp.exp() * logp).sum(-1).mean().item() if c else float("nan"))
    return out


# Inter-GPS predicate -> text; other predicates read as their words, e.g. AreaOf(triangle ABC) -> area of triangle ABC.
LITERALS = {
    "Line": "{all}", "Circle": "circle {0}", "Cirle": "circle {0}", "LengthOf": "{0}", "MeasureOf": "{0}",
    "Equals": "{0} = {1}", "PointLiesOnLine": "{0} lies on {1}", "PointLiesOnCircle": "{0} lies on {1}",
    "Perpendicular": "{0} is perpendicular to {1}", "Parallel": "{0} is parallel to {1}", "Tangent": "{0} is tangent to {1}",
}


def literal_text(literal: str) -> str:
    """An Inter-GPS literal as text, innermost predicate first: Equals(LengthOf(Line(T, R)), x+21) -> TR = x+21."""
    while m := re.search(r"([A-Z]\w*)\(([^()]*)\)", literal):
        name, args = m.group(1), [a.strip() for a in m.group(2).split(",")]
        template = LITERALS.get(name, " ".join(re.findall(r"[A-Z][a-z]*", name)).lower() + " {all}")
        literal = literal[: m.start()] + template.format(*args, all="".join(args)) + literal[m.end() :]
    return literal


def describe_diagram(annotation: dict) -> str:
    lines = [f"Points: {', '.join(annotation['point_instances'])}.", f"Lines: {', '.join(annotation['line_instances'])}."]
    circles = [c for c in annotation["circle_instances"] if c]
    if circles:
        lines.append(f"Circles centered at: {', '.join(circles)}.")
    lines += [literal_text(lf) + "." for lf in annotation["diagram_logic_forms"] if lf.strip()]
    return "Diagram description:\n" + "\n".join(lines)


def diagram_captions(intergps_dir: str) -> dict:
    """Text descriptions of the Geometry3K diagrams from the Inter-GPS gold diagram literals (MIT license), keyed by
    (md5 of the diagram's pixels, question): hiyouga/geometry3k keeps no problem ids and lists them in another order."""
    root = Path(intergps_dir)
    annotations = json.loads((root / "logic_forms" / "diagram_logic_forms_annot.json").read_text())
    captions = {}
    for split in ("train", "val", "test"):
        for problem in (root / split).iterdir():
            question = json.loads((problem / "data.json").read_text())["annotat_text"].strip()
            pixels = hashlib.md5(Image.open(problem / "img_diagram.png").tobytes()).hexdigest()
            captions[(pixels, question)] = describe_diagram(annotations[problem.name])
    return captions


def with_input(ds, input: str, captions: dict | None):
    """Image-dataset rows for --input: keep the image, drop it, or replace it by the diagram description."""
    if input == "caption":
        ds = ds.map(lambda ex: {"question": captions[(hashlib.md5(ex["images"][0].tobytes()).hexdigest(), ex["question"])]
                                + "\n\n" + ex["question"]})
    return ds if input == "image" else ds.remove_columns("images")


def main():
    args = parse_args()
    tokenizer = R.load_tokenizer(args)
    train, test = R.prepare_data(tokenizer, args.dataset, args.max_train_samples, -1, args.data_seed)
    if args.dataset in R.IMAGE_DATASETS:
        captions = diagram_captions(args.intergps_dir) if args.input == "caption" else None
        train, test = (with_input(ds, args.input, captions) for ds in (train, test))

    adapters = sorted(p for pattern in args.adapters for p in glob.glob(pattern))
    if args.adapters and not adapters:
        raise SystemExit(f"No adapter directory matches {args.adapters}")
    adapters = adapters or [None]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["adapter"] for line in out.open()} if out.exists() else set()

    library = None
    if args.own_style:
        lib_path = Path(args.own_style)
        if not lib_path.exists():
            if not args.foreign_ref:
                raise SystemExit("--own_style library missing: pass --foreign_ref to build it")
            model = load_model(args, tokenizer, args.foreign_ref)
            pool = train.map(lambda ex: {"prompt": R.render_question(tokenizer, ex["question"], args.prompt_style)})
            lib_path.parent.mkdir(parents=True, exist_ok=True)
            lib_path.write_text(json.dumps(own_style_library(args, model, tokenizer, pool)))
            del model
            torch.cuda.empty_cache()
        library = {int(k): v for k, v in json.loads(lib_path.read_text()).items()}

    for adapter in adapters:
        if adapter in done:
            continue
        ds = select_prompts(args, tokenizer, train, test, adapter)
        if len(ds) == 0:
            print(f"{adapter}: no prompts to probe (e.g. full_sft has no GRPO bucket), skipped", flush=True)
            continue
        model = load_model(args, tokenizer, adapter)
        if library is not None:
            ex_ids = ds["example_id"]
            mass = own_style_mass(model, tokenizer, list(ds["prompt"]), [library[e]["prefixes"] for e in ex_ids],
                                  args.temperature or 1.0, args.batch_size)
            m0 = [library[e]["m0"] for e in ex_ids]
            with out.open("a") as f:
                f.writelines(
                    json.dumps({"model": args.model, "adapter": adapter, "library": args.own_style, "lam": args.own_style_lam,
                                "example_id": e, "mass": w, "m0": z}) + "\n"
                    for e, w, z in zip(ex_ids, mass, m0)
                )
            print(f"{adapter or 'untrained'}: own-style share = {sum(mass) / max(sum(m0), 1e-12):.2e} "
                  f"(library mass {sum(m0) / len(m0):.2f}) over {len(m0)} prompts", flush=True)
            del model
            torch.cuda.empty_cache()
            continue
        if args.prefix:
            logps = R.prefix_logprobs(model, tokenizer, list(ds["prompt"]), args.prefix, args.temperature or 1.0, args.batch_size)
            with out.open("a") as f:
                f.writelines(
                    json.dumps({"model": args.model, "adapter": adapter, "prefix": args.prefix, "temperature": args.temperature or 1.0,
                                "example_id": ex_id, "logp": lp}) + "\n"
                    for ex_id, lp in zip(ds["example_id"], logps)
                )
            share = sum(math.exp(lp) for lp in logps) / len(logps)
            print(f"{adapter or 'untrained'}: mean P(starts with {args.prefix!r}) = {share:.2e} over {len(logps)} prompts", flush=True)
            del model
            torch.cuda.empty_cache()
            continue
        prefix = args.force_prefix or ""
        if prefix:
            ds = ds.map(lambda ex: {"prompt": ex["prompt"] + prefix})
        rows = [
            {
                "model": args.model, "dataset": args.dataset, "adapter": adapter,
                **({"input": args.input} if args.dataset in R.IMAGE_DATASETS else {}),
                "prompt_style": args.prompt_style, "on": args.on, "temperature": args.temperature, "force_prefix": args.force_prefix,
                "example_id": row["example_id"], "sample": i, "gold": row["gold_answer"],
                "correct": R.answer_is_correct(prefix + text, row["gold_answer"], args.reward == "marked", args.dataset),
                "stopped": stopped, "completion": prefix + text,
            }
            for row, i, text, stopped in generate(args, model, tokenizer, ds)
        ]
        if args.entropy:
            prompts = dict(zip(ds["example_id"], ds["prompt"]))
            entropies = completion_entropy(model, tokenizer, [prompts[r["example_id"]] for r in rows],
                                           [r["completion"] for r in rows], args.temperature or 1.0, max(1, args.batch_size // 4))
            for r, h in zip(rows, entropies):
                r["entropy"] = h
        with out.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"{adapter or 'untrained'} [{args.prompt_style}, {args.on}]: accuracy={acc:.4f} over {len(rows)}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
