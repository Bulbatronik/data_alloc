#!/usr/bin/env python3
"""
SFT/GRPO routing experiment on GSM8K using Hugging Face + TRL.

Supports fixed endpoints, arbitrary exact SFT counts (e.g. random_sft_250),
budget-matched adaptive routing, and per-allocation Weights & Biases logging.

For a full allocation curve across counts/seeds, use run_curve.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Data / answer utilities
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Solve the math problem carefully. Show your reasoning. "
    "End your response with exactly '#### <final numeric answer>'."
)


def extract_gsm8k_gold(answer_text: str) -> str:
    """GSM8K gold solutions conventionally end in '#### <answer>'."""
    if "####" in answer_text:
        return answer_text.rsplit("####", 1)[-1].strip()
    return extract_final_answer(answer_text) or answer_text.strip()


_NUMBER_RE = re.compile(r"-?(?:\d[\d,]*)(?:\.\d+)?")


def extract_final_answer(text: str) -> Optional[str]:
    """Extract a final numeric answer from a model completion."""
    # Prefer the format explicitly requested in the prompt.
    matches = re.findall(r"####\s*(-?(?:\d[\d,]*)(?:\.\d+)?)", text)
    if matches:
        return matches[-1]

    # Fall back to common "final answer is ..." wording.
    matches = re.findall(
        r"final\s+answer(?:\s+is|:)?\s*\$?\s*(-?(?:\d[\d,]*)(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1]

    # Last-resort: final number occurring in the text.
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else None


def normalize_number(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.strip().replace(",", "").replace("$", "")
    x = x.rstrip(".")
    try:
        d = Decimal(x)
        # Decimal normalization without scientific notation.
        out = format(d.normalize(), "f")
        if "." in out:
            out = out.rstrip("0").rstrip(".")
        return "0" if out in {"-0", ""} else out
    except InvalidOperation:
        return x


def answer_is_correct(completion: str, gold: str) -> bool:
    return normalize_number(extract_final_answer(completion)) == normalize_number(gold)


def render_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Generic fallback for base models without a chat template.
    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:\n"


def prepare_gsm8k(tokenizer, max_train_samples: int, max_eval_samples: int, seed: int):
    raw = load_dataset("openai/gsm8k", "main")
    train = raw["train"].shuffle(seed=seed)
    test = raw["test"].shuffle(seed=seed)

    if max_train_samples > 0:
        train = train.select(range(min(max_train_samples, len(train))))
    if max_eval_samples > 0:
        test = test.select(range(min(max_eval_samples, len(test))))

    def convert(ex, idx):
        gold = extract_gsm8k_gold(ex["answer"])
        return {
            "example_id": int(idx),
            "question": ex["question"],
            "gold_trace": ex["answer"],
            "gold_answer": gold,
            "prompt": render_prompt(tokenizer, ex["question"]),
            # Prompt-completion format => TRL SFT computes completion-only loss.
            "completion": ex["answer"],
        }

    train = train.map(convert, with_indices=True, remove_columns=train.column_names)
    test = test.map(convert, with_indices=True, remove_columns=test.column_names)
    return train, test


# ---------------------------------------------------------------------------
# Router: estimate discoverability and gold-trace surprise
# ---------------------------------------------------------------------------

@torch.inference_mode()
def sample_probe_completions(
    model,
    tokenizer,
    prompts: Sequence[str],
    k: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[List[str]]:
    """Return K sampled completions for every prompt."""
    model.eval()
    device = next(model.parameters()).device
    all_groups: List[List[str]] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_width = enc["input_ids"].shape[1]
        generated = out[:, prompt_width:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # HF ordering: K completions for prompt 0, then K for prompt 1, ...
        for j in range(len(batch_prompts)):
            all_groups.append(texts[j * k : (j + 1) * k])

    return all_groups


@torch.inference_mode()
def completion_nll_per_example(
    model,
    tokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    batch_size: int,
    max_length: int,
) -> List[float]:
    """
    Conditional NLL of the gold completion, masking prompt tokens.

    This is a deliberately simple implementation. It is used only as a routing
    feature, not as a training loss.
    """
    model.eval()
    device = next(model.parameters()).device
    results: List[float] = []

    # The mask below assumes every sequence starts at position 0, so pad on the right.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    for start in range(0, len(prompts), batch_size):
        ps = list(prompts[start : start + batch_size])
        cs = list(completions[start : start + batch_size])
        full = [p + c for p, c in zip(ps, cs)]

        enc = tokenizer(
            full,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_lens = [
            min(
                len(tokenizer(p, add_special_tokens=False)["input_ids"]),
                max_length,
            )
            for p in ps
        ]

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Causal shift: token t is predicted from positions < t.
        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)

        for row in range(input_ids.size(0)):
            # label position j predicts original token j+1.
            # First completion token begins at original position prompt_len.
            first_completion_label = max(prompt_lens[row] - 1, 0)
            valid = attention_mask[row, 1:].bool()
            positions = torch.arange(valid.numel(), device=device)
            mask = valid & (positions >= first_completion_label)

            if mask.any():
                results.append(float(token_loss[row][mask].mean().cpu()))
            else:
                results.append(0.0)

    tokenizer.padding_side = old_padding_side
    return results


def build_router_table(
    model,
    tokenizer,
    train_ds: Dataset,
    probe_k: int,
    grpo_group_size: int,
    probe_batch_size: int,
    nll_batch_size: int,
    max_probe_tokens: int,
    sft_max_length: int,
    beta: float,
) -> pd.DataFrame:
    prompts = list(train_ds["prompt"])
    gold_answers = list(train_ds["gold_answer"])
    gold_traces = list(train_ds["completion"])

    print(f"[router] Sampling {probe_k} probe completions for {len(train_ds)} examples...")
    groups = sample_probe_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        k=probe_k,
        batch_size=probe_batch_size,
        max_new_tokens=max_probe_tokens,
        temperature=0.8,
        top_p=0.95,
    )

    correct_counts = [
        sum(answer_is_correct(c, gold) for c in group)
        for group, gold in zip(groups, gold_answers)
    ]
    p = [c / probe_k for c in correct_counts]

    # Probability that a GRPO group contains at least one success AND one failure.
    grpo_value = [
        1.0 - (pi ** grpo_group_size) - ((1.0 - pi) ** grpo_group_size)
        for pi in p
    ]

    print("[router] Computing gold-trace conditional NLL...")
    gold_nll = completion_nll_per_example(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        completions=gold_traces,
        batch_size=nll_batch_size,
        max_length=sft_max_length,
    )

    positive_nll = [x for x in gold_nll if math.isfinite(x) and x > 0]
    median_nll = float(pd.Series(positive_nll).median()) if positive_nll else 1.0
    normalized_nll = [min(x / max(median_nll, 1e-8), 5.0) for x in gold_nll]

    # A gold trace is valuable when the policy rarely succeeds and the gold
    # continuation is still surprising under the current model.
    sft_value = [(1.0 - pi) * nll for pi, nll in zip(p, normalized_nll)]

    # Positive => favor GRPO; negative => favor SFT.
    preference = [r - beta * s for r, s in zip(grpo_value, sft_value)]

    df = pd.DataFrame(
        {
            "example_id": train_ds["example_id"],
            "probe_correct": correct_counts,
            "probe_k": probe_k,
            "p_success": p,
            "gold_nll": gold_nll,
            "gold_nll_normalized": normalized_nll,
            "sft_value": sft_value,
            "grpo_value": grpo_value,
            "grpo_preference": preference,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

def random_partition_count(ds: Dataset, n_sft: int, seed: int) -> Tuple[Dataset, Dataset]:
    """Randomly allocate exactly n_sft examples to SFT; the rest go to GRPO."""
    if not 0 <= n_sft <= len(ds):
        raise ValueError(f"n_sft={n_sft} must be in [0, {len(ds)}].")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return ds.select(indices[:n_sft]), ds.select(indices[n_sft:])


def adaptive_partition_count(
    ds: Dataset,
    router_df: pd.DataFrame,
    n_sft: Optional[int],
) -> Tuple[Dataset, Dataset]:
    """
    Allocate examples with the lowest GRPO-preference scores to SFT.

    If n_sft is None, use the natural decision boundary:
        GRPO when grpo_value > beta * sft_value
        SFT otherwise.

    Otherwise, allocate exactly n_sft examples to SFT. This makes adaptive and
    random curves directly comparable at the same x-axis value.
    """
    scores = router_df["grpo_preference"].to_numpy()

    if n_sft is None:
        sft_idx = [i for i, score in enumerate(scores) if score <= 0.0]
        grpo_idx = [i for i, score in enumerate(scores) if score > 0.0]
    else:
        if not 0 <= n_sft <= len(ds):
            raise ValueError(f"n_sft={n_sft} must be in [0, {len(ds)}].")
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        sft_idx = order[:n_sft]
        grpo_idx = order[n_sft:]

    return ds.select(sft_idx), ds.select(grpo_idx)


def method_metadata(method: str, n_total: int) -> Dict[str, object]:
    """Parse a method name into strategy and exact allocation metadata."""
    if method == "full_sft":
        return {"strategy": "endpoint", "n_sft": n_total, "n_grpo": 0}
    if method == "full_grpo":
        return {"strategy": "endpoint", "n_sft": 0, "n_grpo": n_total}
    if method == "adaptive":
        return {"strategy": "adaptive", "n_sft": None, "n_grpo": None}

    m = re.fullmatch(r"(random|adaptive)_sft_(\d+)", method)
    if m:
        n_sft = int(m.group(2))
        if not 0 <= n_sft <= n_total:
            raise ValueError(f"{method}: n_sft={n_sft} must be in [0, {n_total}].")
        return {
            "strategy": m.group(1),
            "n_sft": n_sft,
            "n_grpo": n_total - n_sft,
        }

    # Backwards-compatible percentage names such as random_25_75.
    m = re.fullmatch(r"(random|adaptive)_(\d+)_(\d+)", method)
    if m:
        sft_pct, grpo_pct = int(m.group(2)), int(m.group(3))
        if sft_pct + grpo_pct != 100:
            raise ValueError(f"Bad percentage method: {method}")
        n_sft = round((sft_pct / 100.0) * n_total)
        return {
            "strategy": m.group(1),
            "n_sft": n_sft,
            "n_grpo": n_total - n_sft,
        }

    raise ValueError(f"Unknown method: {method}")


def method_partition(
    method: str,
    train_ds: Dataset,
    router_df: Optional[pd.DataFrame],
    seed: int,
) -> Tuple[Dataset, Dataset]:
    meta = method_metadata(method, len(train_ds))

    if method == "full_sft":
        return train_ds, train_ds.select([])
    if method == "full_grpo":
        return train_ds.select([]), train_ds
    if method == "adaptive":
        if router_df is None:
            raise ValueError("adaptive routing requested but no router features were built.")
        return adaptive_partition_count(train_ds, router_df, n_sft=None)

    n_sft = int(meta["n_sft"])
    if meta["strategy"] == "random":
        return random_partition_count(train_ds, n_sft, seed)
    if meta["strategy"] == "adaptive":
        if router_df is None:
            raise ValueError("adaptive routing requested but no router features were built.")
        return adaptive_partition_count(train_ds, router_df, n_sft=n_sft)

    raise ValueError(f"Unsupported method: {method}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_lora_config(args) -> Optional[LoraConfig]:
    if args.no_lora:
        return None
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def load_fresh_model(args):
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
    return model


def train_sft(model, tokenizer, ds: Dataset, args, output_dir: Path):
    if len(ds) == 0:
        return model

    # Right padding is preferable for causal-LM training; generation functions
    # below switch back to left padding.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    # Keep only the fields SFTTrainer needs.
    sft_ds = ds.select_columns(["prompt", "completion"])

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.sft_batch_size,
        gradient_accumulation_steps=args.sft_grad_accum,
        learning_rate=args.sft_lr,
        max_length=args.sft_max_length,
        completion_only_loss=True,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=("wandb" if args.wandb_log_train and args.wandb_mode != "disabled" else "none"),
        run_name=getattr(args, "_current_run_name", None),
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_ds,
        processing_class=tokenizer,
        peft_config=make_lora_config(args) if not hasattr(model, "peft_config") else None,
    )
    trainer.train()
    trained_model = trainer.model
    tokenizer.padding_side = old_padding_side
    return trained_model


def gsm8k_reward(completions, gold_answer, **kwargs):
    """Binary exact-answer reward; gold_answer comes from a dataset column."""
    return [
        1.0 if answer_is_correct(completion, gold) else 0.0
        for completion, gold in zip(completions, gold_answer)
    ]


def train_grpo(model, tokenizer, ds: Dataset, args, output_dir: Path):
    if len(ds) == 0:
        return model

    grpo_ds = ds.select_columns(["prompt", "gold_answer"])

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.grpo_epochs,
        per_device_train_batch_size=args.grpo_batch_size,
        gradient_accumulation_steps=args.grpo_grad_accum,
        learning_rate=args.grpo_lr,
        num_generations=args.grpo_group_size,
        max_completion_length=args.grpo_max_completion_length,
        temperature=args.grpo_temperature,
        top_p=args.grpo_top_p,
        beta=args.grpo_kl_beta,
        loss_type=args.grpo_loss_type,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=("wandb" if args.wandb_log_train and args.wandb_mode != "disabled" else "none"),
        run_name=getattr(args, "_current_run_name", None),
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=grpo_ds,
        reward_funcs=gsm8k_reward,
        processing_class=tokenizer,
        peft_config=make_lora_config(args) if not hasattr(model, "peft_config") else None,
    )
    trainer.train()
    return trainer.model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, tokenizer, ds: Dataset, args) -> Dict[str, float]:
    model.eval()
    # SFTTrainer/GRPOTrainer should already place the model on the right device.
    device = next(model.parameters()).device
    correct = 0
    predictions = []

    for start in range(0, len(ds), args.eval_batch_size):
        batch = ds[start : start + args.eval_batch_size]
        prompts = batch["prompt"]
        golds = batch["gold_answer"]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=args.eval_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = out[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for text, gold in zip(texts, golds):
            ok = answer_is_correct(text, gold)
            correct += int(ok)
            predictions.append(
                {
                    "gold": gold,
                    "predicted": extract_final_answer(text),
                    "correct": ok,
                    "completion": text,
                }
            )

    return {
        "accuracy": correct / max(len(ds), 1),
        "correct": correct,
        "n_eval": len(ds),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

DEFAULT_METHODS = [
    "full_sft",
    "full_grpo",
    "random_25_75",
    "random_50_50",
    "random_75_25",
    "adaptive",
    "adaptive_25_75",
    "adaptive_50_50",
    "adaptive_75_25",
]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_init_wandb(args, method: str, n_sft: int, n_grpo: int, strategy: str):
    if args.wandb_mode == "disabled":
        args._current_run_name = None
        return None

    import wandb

    tags = [t for t in args.wandb_tags.split(",") if t] if args.wandb_tags else []
    run_name = f"{strategy}-sft{n_sft}-grpo{n_grpo}-seed{args.seed}"
    args._current_run_name = run_name

    config = {
        "method": method,
        "strategy": strategy,
        "model": args.model,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "n_total": n_sft + n_grpo,
        "num_sft": n_sft,
        "num_grpo": n_grpo,
        "sft_fraction": n_sft / max(n_sft + n_grpo, 1),
        "sft_epochs": args.sft_epochs,
        "sft_lr": args.sft_lr,
        "grpo_epochs": args.grpo_epochs,
        "grpo_lr": args.grpo_lr,
        "grpo_group_size": args.grpo_group_size,
        "grpo_prompts_per_step": args.grpo_batch_size * args.grpo_grad_accum // args.grpo_group_size,
        "router_beta": args.router_beta,
        "probe_k": args.probe_k,
        "grpo_loss_type": args.grpo_loss_type,
        "lora": not args.no_lora,
    }

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or None,
        name=run_name,
        job_type="allocation-run",
        config=config,
        tags=tags,
        mode=args.wandb_mode,
        reinit="finish_previous",
    )
    run.summary["allocation/num_sft"] = n_sft
    run.summary["allocation/num_grpo"] = n_grpo
    run.summary["allocation/sft_fraction"] = config["sft_fraction"]
    return run


def finish_wandb(run, row: Dict[str, object]):
    if run is None:
        return
    run.log(
        {
            "final/accuracy": row["accuracy"],
            "final/correct": row["correct"],
            "final/n_eval": row["n_eval"],
            "allocation/num_sft": row["n_sft"],
            "allocation/num_grpo": row["n_grpo"],
            "allocation/sft_fraction": row["sft_fraction"],
        }
    )
    run.summary["final/accuracy"] = row["accuracy"]
    run.summary["final/correct"] = row["correct"]
    run.summary["final/n_eval"] = row["n_eval"]
    run.finish()


def run(args):
    set_seed(args.seed)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batched generation should left-pad.
    tokenizer.padding_side = "left"

    train_ds, eval_ds = prepare_gsm8k(
        tokenizer,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.data_seed,
    )
    print(f"train={len(train_ds)}, eval={len(eval_ds)}")

    # Resume: keep finished methods from a previous (e.g. preempted) run and skip them.
    results_path = out_root / "results.csv"
    results = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    done = {r["method"] for r in results}
    methods = [m for m in args.methods if m not in done]
    if done:
        print(f"[resume] already finished, skipping: {sorted(done)}")

    needs_router = any(m.startswith("adaptive") for m in methods)
    router_df = None
    router_path = out_root / "router_features.csv"

    if needs_router and router_path.exists():
        print(f"[router] reusing {router_path}")
        router_df = pd.read_csv(router_path)
    elif needs_router:
        print("\n=== Building adaptive router features from the SAME base policy ===")
        router_model = load_fresh_model(args)
        if torch.cuda.is_available():
            router_model = router_model.cuda()

        router_df = build_router_table(
            model=router_model,
            tokenizer=tokenizer,
            train_ds=train_ds,
            probe_k=args.probe_k,
            grpo_group_size=args.grpo_group_size,
            probe_batch_size=args.probe_batch_size,
            nll_batch_size=args.nll_batch_size,
            max_probe_tokens=args.probe_max_new_tokens,
            sft_max_length=args.sft_max_length,
            beta=args.router_beta,
        )
        router_df.to_csv(router_path, index=False)

        natural_sft = int((router_df["grpo_preference"] <= 0).sum())
        print(
            f"[router] natural split: SFT={natural_sft}/{len(router_df)} "
            f"({natural_sft/len(router_df):.1%}), "
            f"GRPO={len(router_df)-natural_sft}/{len(router_df)}"
        )
        del router_model
        cleanup()

    for method in methods:
        print(f"\n{'='*80}\nEXPERIMENT: {method}\n{'='*80}")
        # Reset RNG so a method does not inherit random state consumed by earlier methods.
        set_seed(args.seed)
        exp_dir = out_root / method
        exp_dir.mkdir(parents=True, exist_ok=True)

        sft_ds, grpo_ds = method_partition(method, train_ds, router_df, args.seed)
        parsed = method_metadata(method, len(train_ds))
        strategy = str(parsed["strategy"])
        if method == "adaptive":
            strategy = "adaptive"
        n_sft, n_grpo = len(sft_ds), len(grpo_ds)
        sft_fraction = n_sft / max(len(train_ds), 1)

        print(
            f"partition: SFT={n_sft} ({sft_fraction:.1%}), "
            f"GRPO={n_grpo} ({n_grpo/len(train_ds):.1%})"
        )

        wandb_run = maybe_init_wandb(
            args, method=method, n_sft=n_sft, n_grpo=n_grpo, strategy=strategy
        )

        with open(exp_dir / "partition.json", "w") as f:
            json.dump(
                {
                    "method": method,
                    "n_total": len(train_ds),
                    "n_sft": len(sft_ds),
                    "n_grpo": len(grpo_ds),
                    "sft_example_ids": list(sft_ds["example_id"]),
                    "grpo_example_ids": list(grpo_ds["example_id"]),
                },
                f,
                indent=2,
            )

        model = load_fresh_model(args)

        # Every mixed experiment uses exactly this order:
        # base -> SFT on SFT bucket -> GRPO on disjoint GRPO bucket.
        if len(sft_ds):
            print("[stage 1] SFT")
            if wandb_run is not None:
                wandb_run.log({"stage/sft_started": 1})
            model = train_sft(model, tokenizer, sft_ds, args, exp_dir / "sft")

        if len(grpo_ds):
            print("[stage 2] GRPO")
            if wandb_run is not None:
                wandb_run.log({"stage/grpo_started": 1})
            model = train_grpo(model, tokenizer, grpo_ds, args, exp_dir / "grpo")

        # Final trained weights (the LoRA adapter only, unless --no_lora).
        model.save_pretrained(exp_dir / "final_model")

        print("[eval] greedy exact-match")
        metrics = evaluate(model, tokenizer, eval_ds, args)
        predictions = metrics.pop("predictions")
        pd.DataFrame(predictions).to_json(
            exp_dir / "predictions.jsonl", orient="records", lines=True
        )

        row = {
            "method": method,
            "strategy": strategy,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "n_total": len(train_ds),
            "n_sft": n_sft,
            "n_grpo": n_grpo,
            "sft_fraction": sft_fraction,
            **metrics,
        }
        results.append(row)
        pd.DataFrame(results).to_csv(results_path, index=False)

        print(
            f"{method}: accuracy={row['accuracy']:.4f} "
            f"({row['correct']}/{row['n_eval']})"
        )
        finish_wandb(wandb_run, row)

        del model
        cleanup()

    print("\nFinal results:")
    result_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print(result_df.to_string(index=False))
    result_df.to_csv(results_path, index=False)


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", default="./sft_grpo_routing_runs")
    p.add_argument("--seed", type=int, default=42, help="Training / partition seed.")
    p.add_argument(
        "--data_seed",
        type=int,
        default=1234,
        help="Fixed seed for selecting the dataset subset. Keep constant across training seeds.",
    )
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)

    # Small defaults for a first run. Set -1 to use full splits.
    p.add_argument("--max_train_samples", type=int, default=1000)
    p.add_argument("--max_eval_samples", type=int, default=300)

    # Router.
    p.add_argument("--probe_k", type=int, default=8)
    p.add_argument("--probe_batch_size", type=int, default=8)
    p.add_argument("--probe_max_new_tokens", type=int, default=512)
    p.add_argument("--nll_batch_size", type=int, default=8)
    p.add_argument(
        "--router_beta",
        type=float,
        default=1.0,
        help="Weight on SFT value in grpo_value - beta*sft_value.",
    )

    # SFT.
    p.add_argument("--sft_epochs", type=float, default=1.0)
    p.add_argument("--sft_lr", type=float, default=1e-4)  # LoRA needs ~10x full-FT LR
    p.add_argument("--sft_batch_size", type=int, default=8)
    p.add_argument("--sft_grad_accum", type=int, default=1)
    p.add_argument("--sft_max_length", type=int, default=1024)

    # GRPO.
    p.add_argument("--grpo_epochs", type=float, default=1.0)
    p.add_argument("--grpo_lr", type=float, default=2e-5)  # LoRA needs ~10x full-FT LR
    # batch_size counts completions: 8 completions x 8 accum = 8 prompts per optimizer step.
    p.add_argument("--grpo_batch_size", type=int, default=8)
    p.add_argument("--grpo_grad_accum", type=int, default=8)
    p.add_argument("--grpo_group_size", type=int, default=8)
    p.add_argument("--grpo_max_completion_length", type=int, default=512)
    p.add_argument("--grpo_temperature", type=float, default=0.8)
    p.add_argument("--grpo_top_p", type=float, default=0.95)
    p.add_argument("--grpo_kl_beta", type=float, default=0.0)
    p.add_argument(
        "--grpo_loss_type",
        default="grpo",
        choices=["grpo", "dr_grpo", "dapo", "bnpo"],
        help="TRL currently defaults to DAPO; this experiment defaults explicitly to vanilla GRPO.",
    )

    # Evaluation.
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--eval_max_new_tokens", type=int, default=512)

    # Weights & Biases.
    p.add_argument("--wandb_project", default="sft-grpo-routing")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_group", default="", help="Defaults to the model name.")
    p.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    p.add_argument(
        "--wandb_log_train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log SFT/GRPO Trainer metrics into the allocation run.",
    )
    p.add_argument(
        "--wandb_tags",
        default="gsm8k,sft-grpo-routing",
        help="Comma-separated W&B tags.",
    )

    # LoRA / hardware.
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--logging_steps", type=int, default=10)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16.")

    # One W&B group per model, e.g. "Qwen2.5-0.5B".
    args.wandb_group = args.wandb_group or args.model.split("/")[-1]

    # TRL GRPO requirement: effective batch must be divisible by num_generations.
    effective_grpo_batch = args.grpo_batch_size * args.grpo_grad_accum
    if effective_grpo_batch % args.grpo_group_size != 0:
        raise ValueError(
            "For single-process training, grpo_batch_size * grpo_grad_accum "
            f"({effective_grpo_batch}) must be divisible by grpo_group_size "
            f"({args.grpo_group_size})."
        )

    run(args)
