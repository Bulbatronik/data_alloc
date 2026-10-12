#!/usr/bin/env python3
"""
SFT/GRPO routing experiment on GSM8K, MATH or Geometry3K (with a VLM) using Hugging Face + TRL.

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
import sympy
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    ProcessorMixin,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from trl.data_utils import prepare_multimodal_messages

from modes import MODES, classify


# ---------------------------------------------------------------------------
# Data / answer utilities
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Solve the math problem carefully. Show your reasoning. "
    "End your response with exactly '#### <final numeric answer>'."
)

# Meta's 0-shot CoT template for Llama 3 on MATH (the model's native math format).
LLAMA_MATH_PROMPT = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n"
    "- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem.\n\n"
    "Problem: {question}"
)

# name -> (system prompt, user template)
PROMPT_STYLES = {
    "train": (SYSTEM_PROMPT, "{question}"),
    "boxed": ("Please reason step by step, and put your final answer within \\boxed{}.", "{question}"),
    "boxed_user": (None, "{question}\nPlease reason step by step, and put your final answer within \\boxed{}."),
    "plain": (None, "{question}"),
    "llama_math": (None, LLAMA_MATH_PROMPT),
}


def extract_gsm8k_gold(answer_text: str) -> str:
    """GSM8K gold solutions conventionally end in '#### <answer>'."""
    if "####" in answer_text:
        return answer_text.rsplit("####", 1)[-1].strip()
    return extract_final_answer(answer_text) or answer_text.strip()


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...}, allowing nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for i in range(start + len("\\boxed{"), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start + len("\\boxed{") : i]
            depth -= 1
    return None


def numeric_answer(answer: Optional[str]) -> Optional[str]:
    """The answer if it is a plain number (MATH writes e.g. '10,\\!000'), else None."""
    if answer is None:
        return None
    cleaned = answer.replace("\\!", "").replace(",", "").replace("$", "").strip()
    return cleaned if re.fullmatch(r"-?\d+(\.\d+)?", cleaned) else None


_NUM = r"-?(?:\d[\d,]*)(?:\.\d+)?"
_ANSWER_PATTERNS = [
    rf"####\s*({_NUM})",  # the format the prompt asks for
    rf"\\boxed\{{\s*\\?\$?\s*({_NUM})",
    rf"final\s+answer(?:\s+is|:)?\s*\$?\s*({_NUM})",
]
# An explicitly marked final answer only (no last-number fallback). Also accepts the prompt's placeholder copied
# verbatim ('#### <final numeric answer> 15'), '$' and markdown bold, and 'answer is: N'.
_MARKED_PATTERNS = [
    rf"####\s*(?:<final numeric answer>)?[\s:]*\$?\s*\**\s*({_NUM})",
    rf"\\boxed\{{\s*\\?\$?\s*({_NUM})",
    rf"(?:final\s+)?answer\s*(?:is|:)\s*:?\s*\$?\s*\**\s*({_NUM})",
]


def extract_final_answer(text: str, marked: bool = False) -> Optional[str]:
    """Extract a final numeric answer from a model completion. lenient (default) falls back to the last number;
    marked accepts only an explicitly marked answer."""
    # Base-model prompts end in "Answer:"; ignore any follow-up question the model invents after answering.
    text = text.split("\nQuestion:")[0]
    for pattern in _MARKED_PATTERNS if marked else _ANSWER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1]
    if marked:
        return None
    nums = re.findall(_NUM, text)
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


def extract_expression(text: str, marked: bool = False) -> Optional[str]:
    """Free-form final answer: the last \\boxed{...}, else a '#### ...' line. lenient also takes '(final) answer is ...',
    then the last number."""
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    lines = re.findall(r"####\s*(.+)", text)
    if not lines and not marked:
        lines = re.findall(r"answer\s*(?:is|:)\s*(.+)", text, flags=re.IGNORECASE)
    if lines:
        return lines[-1].strip().rstrip(".")
    if marked:
        return None
    nums = re.findall(_NUM, text)
    return nums[-1] if nums else None


# Applied in order to a LaTeX answer before the brace rewrites in latex_to_sympy.
_LATEX_REWRITES = [
    (r"\\(?:text|mathrm|mbox)\s*\{[^{}]*\}(?:\^\s*\{?[23]\}?)?", ""),  # units
    (r"\^\s*\{?\s*\\circ\s*\}?|°|\\degree|\\%|\$|\\left|\\right|\\displaystyle|\\[,;:! ]", ""),
    (r"\s+", ""),
    (r"(?<=[\d)}])(?:sq(?:uare)?)?(?:units?|cm|mm|km|m|in|inches|ft|feet|yd|degrees?)(?:\^\{?[23]\}?|²|³)?$", ""),
    (r"(\d),(\d{3})(?!\d)", r"\1\2"),
    (r"\\[dt]frac", r"\\frac"),
    (r"(?<![\d.^])(\d+)\\frac\{(\d+)\}\{(\d+)\}", r"(\1+\2/\3)"),  # mixed number: 4\frac{4}{11}
    (r"\\frac(\d)(\d)", r"(\1/\2)"),
    (r"\\sqrt(\d+|[a-zA-Z])", r" sqrt(\1)"),
    (r"\\pi", " pi "),
    (r"\\cdot|\\times", "*"),
    (r"\\div|:", "/"),  # ratios: 2:3 = 2/3
    (r"\^", "**"),
]
_SYMPY_NAMES = {**{c: sympy.Symbol(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"},
                "pi": sympy.pi, "sqrt": sympy.sqrt}


def latex_to_sympy(answer: str):
    """sympy expression of a LaTeX answer ('2 \\sqrt { 221 }', '\\frac{15}{2}\\sqrt3', '12\\pi', '60^\\circ'), or None."""
    s = answer
    for pattern, repl in _LATEX_REWRITES:
        s = re.sub(pattern, repl, s)
    while True:  # innermost groups first: \frac{a}{b} -> ((a)/(b)), \sqrt{a} -> sqrt(a), other {a} -> (a)
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        t = re.sub(r"\\sqrt\{([^{}]*)\}", r" sqrt(\1)", t)
        if t == s:
            t = re.sub(r"(?<!\\frac)(?<!\\sqrt)(?<!\})\{([^{}]*)\}", r"(\1)", s)
        if t == s:
            break
        s = t
    # Only numbers, operators, single-letter variables, pi and sqrt reach the parser, with at most one small power
    # (sympy evaluates powers exactly, so 9^{9^{9^9}} would never finish).
    if (not re.fullmatch(r"[a-zA-Z0-9.+\-*/() ]+", s) or len(s) > 200 or s.count("**") > 1
            or re.search(r"\*\*\(*\d{3}", s) or not set(re.findall(r"[a-zA-Z]+", s)) <= set(_SYMPY_NAMES)):
        return None
    try:
        return parse_expr(s, local_dict=_SYMPY_NAMES,
                          transformations=standard_transformations + (implicit_multiplication_application,))
    except Exception:  # parse_expr raises many error types on malformed input
        return None


def _plain(answer: str) -> str:
    return re.sub(r"\s|\$|\\left|\\right|\\[,;!]", "", answer).lower().rstrip(".")


def expressions_match(predicted: Optional[str], gold: str) -> bool:
    """Equal as text, or as numbers (1e-4 relative tolerance), or symbolically (answers with variables)."""
    if predicted is None:
        return False
    predicted = predicted.replace("\\approx", "=")
    if "=" in predicted and "=" not in gold:  # 'x = 5'
        predicted = predicted.rsplit("=", 1)[1]
    if _plain(predicted) == _plain(gold):
        return True
    a, b = latex_to_sympy(predicted), latex_to_sympy(gold)
    if a is None or b is None:
        return False
    try:
        if a.free_symbols or b.free_symbols:
            return sympy.simplify(a - b) == 0
        return math.isclose(float(a), float(b), rel_tol=1e-4)
    except (TypeError, ValueError):  # complex or undefined values
        return False


def extract_answer(text: str, marked: bool = False, dataset: str = "gsm8k") -> Optional[str]:
    return extract_expression(text, marked) if dataset == "geometry3k" else extract_final_answer(text, marked)


def answer_is_correct(completion: str, gold: str, marked: bool = False, dataset: str = "gsm8k") -> bool:
    predicted = extract_answer(completion, marked, dataset)
    if dataset == "geometry3k":
        return expressions_match(predicted, gold)
    return normalize_number(predicted) == normalize_number(gold)


STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>")


def text_tokenizer(tokenizer):
    """The tokenizer itself, or the one inside a VLM processor."""
    return tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer


def stop_token_ids(tokenizer) -> List[int]:
    """EOS plus end-of-turn tokens: base Qwen's EOS is <|endoftext|>, but its chat turns end in <|im_end|>."""
    tokenizer = text_tokenizer(tokenizer)
    vocab = tokenizer.get_vocab()
    return sorted({tokenizer.eos_token_id} | {vocab[t] for t in STOP_TOKENS if t in vocab})


def chat_messages(question: str, system_prompt: Optional[str]) -> List[dict]:
    messages = [{"role": "user", "content": question}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def render_prompt(tokenizer, question: str, system_prompt: Optional[str] = SYSTEM_PROMPT) -> str:
    messages = chat_messages(question, system_prompt)
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Llama 3 templates insert today's date; pin it so prompts don't change
            # between days (ignored by templates that don't use it, e.g. Qwen).
            date_string="26 Jul 2024",
        )
        # Some templates (e.g. Llama 3) already contain the BOS token, and the
        # tokenizer adds it again when the prompt is tokenized. Keep only one.
        bos = tokenizer.bos_token
        if bos and text.startswith(bos) and tokenizer("x")["input_ids"][:1] == [tokenizer.bos_token_id]:
            text = text[len(bos):]
        return text
    # Generic fallback for base models without a chat template.
    return f"{system_prompt or ''}\n\nQuestion: {question}\nAnswer:\n"


def load_problems(dataset: str, split: str) -> Dataset:
    """Rows with `question`, `gold_trace` (SFT target) and `gold_answer`, plus `images` for image datasets."""
    if dataset == "geometry3k":
        # Diagram + question with a free-form LaTeX answer (graded by expressions_match). There are no gold
        # solutions: the 'gold' SFT target is the boxed answer alone, so pass teacher traces with --sft_traces.
        ds = load_dataset("hiyouga/geometry3k", split=split)
        return ds.map(
            lambda ex: {
                "question": ex["problem"].replace("<image>", "").strip(),
                "gold_trace": f"\\boxed{{{ex['answer']}}}",
                "gold_answer": ex["answer"],
            },
            remove_columns=["problem", "answer"],
        )

    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        return ds.map(
            lambda ex: {
                "question": ex["question"],
                "gold_trace": ex["answer"],
                "gold_answer": extract_gsm8k_gold(ex["answer"]),
            },
            remove_columns=ds.column_names,
        )

    # Evaluation-only, out-of-distribution sets (the split argument is ignored).
    if dataset == "svamp":
        ds = concatenate_datasets([load_dataset("ChilleD/SVAMP", split=s) for s in ("train", "test")])
        return ds.map(
            lambda ex: {"question": ex["question_concat"], "gold_trace": ex["Equation"], "gold_answer": str(ex["Answer"])},
            remove_columns=ds.column_names,
        )
    if dataset == "gsm_plus":
        ds = load_dataset("qintongli/GSM-Plus", split="testmini")
        ds = ds.filter(lambda ex: numeric_answer(str(ex["answer"])) is not None)  # drops "critical thinking" items
        return ds.map(
            lambda ex: {"question": ex["question"], "gold_trace": ex["solution"], "gold_answer": numeric_answer(str(ex["answer"]))},
            remove_columns=ds.column_names,
        )

    # MATH levels 1-3 with a plain numeric answer, so the same numeric grader applies.
    # Gold traces get the '#### <answer>' line our prompt asks for, as GSM8K traces have.
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split=split)
    ds = ds.filter(lambda ex: ex["level"] in {"Level 1", "Level 2", "Level 3"})
    ds = ds.map(lambda ex: {"gold_answer": numeric_answer(last_boxed(ex["solution"]))})
    ds = ds.filter(lambda ex: ex["gold_answer"] is not None)
    return ds.map(
        lambda ex: {
            "question": ex["problem"],
            "gold_trace": f"{ex['solution']}\n#### {ex['gold_answer']}",
            "gold_answer": ex["gold_answer"],
        },
        remove_columns=ds.column_names,
    )


def correct_traces(path: str) -> Dict[int, str]:
    """First correct completion per problem in a probe.py file."""
    probes = pd.read_json(path, lines=True)
    return probes[probes["correct"]].groupby("example_id")["completion"].first().to_dict()


def with_sft_targets(ds: Dataset, source: str) -> Dataset:
    """
    Choose the SFT completions:
      gold             the dataset's traces as they are
      stripped         gold traces without the <<calc>> annotations
      <probe.jsonl>    the first correct completion per problem (problems without one are dropped from SFT)
      mix:<k>:<probe>  gold traces, except k problems that get their probe completion instead
    """
    if source == "gold" or len(ds) == 0:
        return ds
    if source == "stripped":
        return ds.map(lambda ex: {"completion": re.sub(r"<<[^<>]*>>", "", ex["completion"])})
    if source.startswith("mix:"):
        _, k, path = source.split(":", 2)
        traces = correct_traces(path)
        swap = set([i for i in ds["example_id"] if i in traces][: int(k)])
        return ds.map(lambda ex: {"completion": traces[ex["example_id"]] if ex["example_id"] in swap else ex["completion"]})
    traces = correct_traces(source)
    ds = ds.filter(lambda ex: ex["example_id"] in traces)
    return ds.map(lambda ex: {"completion": traces[ex["example_id"]]})


def render_question(tokenizer, question: str, style: str = "train"):
    """Prompt text, or chat messages for a VLM processor (the image is added from the `images` column)."""
    system, template = PROMPT_STYLES[style]
    question = template.replace("{question}", question)
    if isinstance(tokenizer, ProcessorMixin):
        return chat_messages(question, system)
    return render_prompt(tokenizer, question, system)


def encode_prompts(tokenizer, prompts: Sequence, images: Optional[Sequence] = None):
    """Left-padded model inputs for text prompts, or for chat-message prompts and their images (VLM processor)."""
    if isinstance(prompts[0], str):
        return tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
    messages = [prepare_multimodal_messages(p, list(imgs)) for p, imgs in zip(prompts, images or [[]] * len(prompts))]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        processor_kwargs={"padding": True},
    )


def prepare_data(
    tokenizer, dataset: str, max_train_samples: int, max_eval_samples: int, seed: int, prompt_style: str = "train"
):
    train = load_problems(dataset, "train").shuffle(seed=seed)
    test = load_problems(dataset, "test").shuffle(seed=seed)

    if max_train_samples > 0:
        train = train.select(range(min(max_train_samples, len(train))))
    if max_eval_samples > 0:
        test = test.select(range(min(max_eval_samples, len(test))))

    def convert(ex, idx):
        return {
            "example_id": int(idx),
            "prompt": render_question(tokenizer, ex["question"], prompt_style),
            # Prompt-completion format => TRL SFT computes completion-only loss.
            "completion": ex["gold_trace"],
        }

    return train.map(convert, with_indices=True), test.map(convert, with_indices=True)


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
    images: Optional[Sequence] = None,
) -> List[List[str]]:
    """Return K sampled completions for every prompt (with its images, if given)."""
    model.eval()
    device = next(model.parameters()).device
    all_groups: List[List[str]] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        enc = encode_prompts(tokenizer, batch_prompts, images[start : start + batch_size] if images else None).to(device)

        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,  # as in TRL's GRPO rollouts
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=text_tokenizer(tokenizer).pad_token_id,
            eos_token_id=stop_token_ids(tokenizer),
        )

        prompt_width = enc["input_ids"].shape[1]
        generated = out[:, prompt_width:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # HF ordering: K completions for prompt 0, then K for prompt 1, ...
        for j in range(len(batch_prompts)):
            all_groups.append(texts[j * k : (j + 1) * k])

    return all_groups


@torch.inference_mode()
def token_logprobs(model, tokenizer, prompts, continuations, temperature: float, batch_size: int):
    """Teacher-forced per-token log-probabilities of each continuation (token ids) after its prompt."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(prompts), batch_size):
        conts = continuations[start : start + batch_size]
        seqs = [tokenizer(p)["input_ids"] + c for p, c in zip(prompts[start : start + batch_size], conts)]
        ids = torch.full((len(seqs), max(map(len, seqs))), tokenizer.pad_token_id)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq)
            mask[i, : len(seq)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        for i, (seq, c) in enumerate(zip(seqs, conts)):
            n, k = len(seq), len(c)
            logp = (logits[i, n - k - 1 : n - 1].float() / temperature).log_softmax(-1)
            out.append(logp[torch.arange(k), torch.tensor(c)].tolist())
    return out


def prefix_logprobs(model, tokenizer, prompts, prefix: str, temperature: float, batch_size: int):
    """Teacher-forced log-probability, per prompt, that the completion starts with `prefix`."""
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    return [sum(lp) for lp in token_logprobs(model, tokenizer, prompts, [prefix_ids] * len(prompts), temperature, batch_size)]


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
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_lens = [
            min(
                len(tokenizer(p)["input_ids"]),
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


def easy_partition_count(ds: Dataset, router_df: pd.DataFrame, n_sft: int) -> Tuple[Dataset, Dataset]:
    """The opposite routing direction: the easiest problems (highest probe pass rate, then most familiar
    gold trace) go to SFT, the rest to GRPO."""
    order = router_df.sort_values(["p_success", "gold_nll"], ascending=[False, True], kind="stable").index.tolist()
    return ds.select(order[:n_sft]), ds.select(order[n_sft:])


def method_metadata(method: str, n_total: int) -> Dict[str, object]:
    """Parse a method name into strategy and exact allocation metadata."""
    if method == "full_sft":
        return {"strategy": "endpoint", "n_sft": n_total, "n_grpo": 0}
    if method == "full_grpo":
        return {"strategy": "endpoint", "n_sft": 0, "n_grpo": n_total}
    if method == "adaptive":
        return {"strategy": "adaptive", "n_sft": None, "n_grpo": None}
    if method == "listed_sft":
        return {"strategy": "listed", "n_sft": None, "n_grpo": None}

    m = re.fullmatch(r"(random|adaptive|easy)_sft_(\d+)", method)
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
    sft_ids: Optional[set] = None,
) -> Tuple[Dataset, Dataset]:
    meta = method_metadata(method, len(train_ds))
    if method == "listed_sft":
        if sft_ids is None:
            raise ValueError("listed_sft needs --sft_ids.")
        return (train_ds.filter(lambda ex: ex["example_id"] in sft_ids),
                train_ds.filter(lambda ex: ex["example_id"] not in sft_ids))

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
    if meta["strategy"] in ("adaptive", "easy") and router_df is None:
        raise ValueError(f"{meta['strategy']} routing requested but no router features were built.")
    if meta["strategy"] == "adaptive":
        return adaptive_partition_count(train_ds, router_df, n_sft=n_sft)
    if meta["strategy"] == "easy":
        return easy_partition_count(train_ds, router_df, n_sft=n_sft)

    raise ValueError(f"Unsupported method: {method}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

IMAGE_DATASETS = {"geometry3k"}
# LoRA on the language model only: the vision tower and its projector stay frozen.
VISION_MODULES = r".*\.(visual|vision_tower|vision_model|multi_modal_projector)\..*"


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
        exclude_modules=VISION_MODULES if args.dataset in IMAGE_DATASETS else None,
    )


def load_tokenizer(args):
    """Left-padding tokenizer, or for image datasets the model's processor (TRL then trains it as a VLM)."""
    if args.dataset in IMAGE_DATASETS:
        processor = AutoProcessor.from_pretrained(args.model)
        processor.tokenizer.padding_side = "left"
        return processor
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batched generation should left-pad.
    tokenizer.padding_side = "left"
    return tokenizer


def load_fresh_model(args):
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    auto_class = AutoModelForImageTextToText if args.dataset in IMAGE_DATASETS else AutoModelForCausalLM
    model = auto_class.from_pretrained(
        args.model,
        dtype=dtype,
    )
    # Drop the vendor's sampling defaults (e.g. Qwen-Instruct's repetition_penalty=1.1),
    # which transformers otherwise applies to every generate() call that doesn't set them.
    model.generation_config = GenerationConfig()
    if args.gradient_checkpointing:
        model.config.use_cache = False
    return model


def train_sft(model, tokenizer, ds: Dataset, args, output_dir: Path):
    if len(ds) == 0:
        return model

    # Right padding is preferable for causal-LM training; generation functions
    # below switch back to left padding.
    old_padding_side = text_tokenizer(tokenizer).padding_side
    text_tokenizer(tokenizer).padding_side = "right"

    # Keep only the fields SFTTrainer needs.
    if "images" in ds.column_names:  # TRL's VLM collator wants the completion as an assistant message too
        sft_ds = ds.select_columns(["prompt", "completion", "images"])
        sft_ds = sft_ds.map(lambda ex: {"completion": [{"role": "assistant", "content": ex["completion"]}]})
    else:
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
    trained_model.save_pretrained(output_dir)  # adapter after the SFT stage
    text_tokenizer(tokenizer).padding_side = old_padding_side
    return trained_model


def make_reward(log_dir: Path, stop_ids: Sequence[int], marked: bool = False, dataset: str = "gsm8k",
                resuming: bool = False, needs_stop: bool = False, censor: Optional[str] = None,
                log_rollouts: bool = False):
    """Binary exact-answer reward. Appends each step's rollout mode mix to log_dir/rollout_modes.csv and, with
    log_rollouts, every rollout to log_dir/rollouts.jsonl.

    needs_stop: an answer earns reward only if the completion ended (EOS) before the length limit.
    censor:     rollouts matching this regex get reward None, which TRL drops from the group baseline and gives
                zero advantage, so the matched behaviour is never reinforced or penalised (censored transport).
    """
    modes_path, rollouts_path = log_dir / "rollout_modes.csv", log_dir / "rollouts.jsonl"
    if not resuming:
        modes_path.unlink(missing_ok=True)
        rollouts_path.unlink(missing_ok=True)
    censor_re = re.compile(censor) if censor else None

    def gsm8k_reward(completions, gold_answer, completion_ids=None, trainer_state=None, **kwargs):
        # Conversational (VLM) prompts get completions as [{"role": "assistant", "content": ...}].
        completions = [c if isinstance(c, str) else c[0]["content"] for c in completions]
        ids = completion_ids or [[]] * len(completions)
        stopped = [bool(t) and t[-1] in stop_ids for t in ids]
        correct = [answer_is_correct(c, g, marked, dataset) for c, g in zip(completions, gold_answer)]
        rewards = [1.0 if ok and (s or not needs_stop) else 0.0 for ok, s in zip(correct, stopped)]
        censored = [bool(censor_re and censor_re.search(c)) for c in completions]
        step = trainer_state.global_step if trainer_state else None

        modes = [classify(c) for c in completions]
        row = {"step": step, "n": len(rewards), "censored": sum(censored)}
        for mode in MODES:
            hits = [r for r, m in zip(rewards, modes) if m == mode]
            row[f"w_{mode}"] = len(hits) / len(rewards)
            row[f"p_{mode}"] = sum(hits) / len(hits) if hits else float("nan")
        pd.DataFrame([row]).to_csv(modes_path, mode="a", header=not modes_path.exists(), index=False)

        if log_rollouts:
            example_ids = kwargs.get("example_id") or [None] * len(completions)
            with open(rollouts_path, "a") as f:
                for i, c in enumerate(completions):
                    f.write(json.dumps({"step": step, "example_id": example_ids[i], "mode": classify(c, split_native=True),
                                        "reward": rewards[i], "correct": correct[i], "stopped": stopped[i],
                                        "censored": censored[i], "n_tokens": len(ids[i]), "completion": c}) + "\n")
        return [None if cut else r for r, cut in zip(rewards, censored)]

    return gsm8k_reward


class PrefixPanel(TrainerCallback):
    """Teacher-forced log p(completion starts with each prefix) on fixed prompts, at step 0 and every `every` steps.
    Writes one row per (step, prefix) with the per-prompt log-probabilities to `path` (jsonl)."""

    def __init__(self, tokenizer, prompts: Sequence[str], prefixes: Sequence[str], every: int, path: Path,
                 resuming: bool, batch_size: int = 32):
        self.tokenizer, self.prompts, self.prefixes = tokenizer, list(prompts), list(prefixes)
        self.every, self.path, self.batch_size = every, path, batch_size
        if not resuming:
            path.unlink(missing_ok=True)

    def measure(self, model, step: int):
        was_training = model.training
        model.eval()
        with open(self.path, "a") as f:
            for prefix in self.prefixes:
                logp = prefix_logprobs(model, self.tokenizer, self.prompts, prefix, 1.0, self.batch_size)
                f.write(json.dumps({"step": step, "prefix": prefix, "logp": logp}) + "\n")
        model.train(was_training)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 and not self.path.exists():
            self.measure(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every == 0:
            self.measure(model, state.global_step)


def train_grpo(model, tokenizer, ds: Dataset, args, output_dir: Path, panel_prompts: Sequence[str] = ()):
    if len(ds) == 0:
        return model

    grpo_ds = ds.select_columns(["prompt", "gold_answer", "example_id"] + (["images"] if "images" in ds.column_names else []))
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.grpo_epochs,
        max_steps=args.grpo_steps,  # > 0 overrides epochs and cycles the bucket
        lr_scheduler_type=args.grpo_lr_scheduler,
        warmup_steps=args.grpo_warmup_steps,
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
        save_strategy="steps" if args.grpo_save_steps else "no",
        save_steps=args.grpo_save_steps or 500,
        report_to=("wandb" if args.wandb_log_train and args.wandb_mode != "disabled" else "none"),
        run_name=getattr(args, "_current_run_name", None),
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        seed=args.grpo_seed,
        generation_kwargs={"eos_token_id": stop_token_ids(tokenizer)},
    )

    resume = get_last_checkpoint(str(output_dir))  # a preempted run continues from its last saved step
    reward = make_reward(output_dir, stop_token_ids(tokenizer), marked=args.reward == "marked", dataset=args.dataset,
                         resuming=resume is not None, needs_stop=args.reward_needs_stop, censor=args.censor,
                         log_rollouts=args.log_rollouts)
    callbacks = []
    if args.panel_prefix and panel_prompts:
        callbacks.append(PrefixPanel(text_tokenizer(tokenizer), panel_prompts, args.panel_prefix, args.panel_every,
                                     output_dir / "panel.jsonl", resuming=resume is not None))
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=grpo_ds,
        reward_funcs=reward,
        processing_class=tokenizer,
        peft_config=make_lora_config(args) if not hasattr(model, "peft_config") else None,
        callbacks=callbacks,
    )
    if resume:
        print(f"[grpo] resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.model.save_pretrained(output_dir)  # adapter after the GRPO stage
    return trainer.model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, tokenizer, ds: Dataset, args) -> Dict[str, float]:
    """Greedy accuracy, or with --eval_temperature > 0 the mean accuracy of --eval_k samples per question (pass@1).
    Greedy batched decoding is not reproducible across batch sizes (±1-2.6 points on the same weights); sampled
    pass@1 averages that out."""
    model.eval()
    # SFTTrainer/GRPOTrainer should already place the model on the right device.
    device = next(model.parameters()).device
    marked = getattr(args, "reward", "lenient") == "marked"
    dataset = getattr(args, "dataset", "gsm8k")
    temperature = getattr(args, "eval_temperature", 0.0)
    k = getattr(args, "eval_k", 1) if temperature > 0 else 1
    per_batch = max(1, args.eval_batch_size // k)
    sampling = ({"do_sample": True, "temperature": temperature, "top_p": args.grpo_top_p, "top_k": 0}
                if temperature > 0 else {"do_sample": False})
    correct = correct_other = 0
    predictions = []

    for start in range(0, len(ds), per_batch):
        batch = ds[start : start + per_batch]
        enc = encode_prompts(tokenizer, batch["prompt"], batch.get("images")).to(device)

        out = model.generate(
            **enc,
            **sampling,
            num_return_sequences=k,
            max_new_tokens=args.eval_max_new_tokens,
            pad_token_id=text_tokenizer(tokenizer).pad_token_id,
            eos_token_id=stop_token_ids(tokenizer),
        )
        generated = out[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        stopped = torch.isin(generated, torch.tensor(stop_token_ids(tokenizer), device=device)).any(dim=1)

        # HF ordering: k completions for question 0, then k for question 1, ...
        for j, (text, did_stop) in enumerate(zip(texts, stopped.tolist())):
            gold = batch["gold_answer"][j // k]
            ok = answer_is_correct(text, gold, marked, dataset)
            correct += int(ok)
            correct_other += int(answer_is_correct(text, gold, not marked, dataset))
            predictions.append(
                {
                    "gold": gold,
                    "sample": j % k,
                    "predicted": extract_answer(text, marked, dataset),
                    "correct": ok,
                    "stopped": did_stop,
                    "completion": text,
                }
            )

    n = max(len(ds) * k, 1)
    return {
        "accuracy": correct / n,  # graded like the RL reward (--reward)
        f"accuracy_{'lenient' if marked else 'marked'}": correct_other / n,
        "correct": correct,
        "n_eval": len(ds) * k,  # graded answers: k per question
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

    tags = [args.dataset] + [t for t in args.wandb_tags.split(",") if t]
    run_name = f"{args.dataset}-{strategy}-sft{n_sft}-grpo{n_grpo}-seed{args.seed}"
    if args.grpo_seed != args.seed:
        run_name += f"-grposeed{args.grpo_seed}"
    if args.sft_traces.startswith("mix:"):
        run_name += f"-mix{args.sft_traces.split(':')[1]}"
    if args.sft_traces != "gold":
        run_name += f"-{Path(args.sft_traces).stem}"
    if args.sft_ids:
        run_name += f"-{Path(args.sft_ids).stem}"
    if args.prompt_style != "train":
        run_name += f"-{args.prompt_style}"
    if args.sft_epochs != 1:
        run_name += f"-sftep{args.sft_epochs:g}"
    if args.grpo_steps > 0:
        run_name += f"-grpo{args.grpo_steps}steps"
    if args.reward != "lenient":
        run_name += f"-{args.reward}"
    if args.reward_needs_stop:
        run_name += "-needstop"
    if args.censor:
        run_name += "-censored"
    if args.grpo_loss_type != "grpo":
        run_name += f"-{args.grpo_loss_type}"
    args._current_run_name = run_name

    config = {
        "dataset": args.dataset,
        "method": method,
        "strategy": strategy,
        "model": args.model,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "n_total": n_sft + n_grpo,
        "num_sft": n_sft,
        "num_grpo": n_grpo,
        "sft_fraction": n_sft / max(n_sft + n_grpo, 1),
        "sft_traces": args.sft_traces,
        "prompt_style": args.prompt_style,
        "sft_epochs": args.sft_epochs,
        "sft_lr": args.sft_lr,
        "grpo_epochs": args.grpo_epochs,
        "grpo_lr": args.grpo_lr,
        "grpo_group_size": args.grpo_group_size,
        "grpo_prompts_per_step": args.grpo_batch_size * args.grpo_grad_accum // args.grpo_group_size,
        "router_beta": args.router_beta,
        "probe_k": args.probe_k,
        "grpo_loss_type": args.grpo_loss_type,
        "grpo_steps": args.grpo_steps,
        "grpo_lr_scheduler": args.grpo_lr_scheduler,
        "reward": args.reward,
        "reward_needs_stop": args.reward_needs_stop,
        "censor": args.censor,
        "eval_temperature": args.eval_temperature,
        "eval_k": args.eval_k,
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
    final = {
        "final/accuracy": row["accuracy"],
        "final/correct": row["correct"],
        "final/n_eval": row["n_eval"],
        "base/accuracy": row["base_accuracy"],
        "after_sft/accuracy": row["after_sft_accuracy"],
        "gain/sft_vs_base": row["sft_gain"],
        "gain/grpo_vs_start": row["grpo_gain"],
        "allocation/num_sft": row["n_sft"],
        "allocation/num_grpo": row["n_grpo"],
        "allocation/sft_fraction": row["sft_fraction"],
    }
    run.log(final)
    run.summary.update(final)
    run.finish()


def run(args):
    set_seed(args.seed)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args)

    train_ds, eval_ds = prepare_data(
        tokenizer,
        prompt_style=args.prompt_style,
        dataset=args.dataset,
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

    needs_router = any(m.startswith(("adaptive", "easy")) for m in methods)
    if needs_router and args.dataset in IMAGE_DATASETS:
        raise ValueError("adaptive/easy routing is text-only (the router's trace NLL takes no images).")
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

    # Untrained model, evaluated once per seed folder (reference for the SFT/GRPO gains).
    base_path = out_root / "base_eval.json"
    if methods and not base_path.exists():
        print("\n=== Evaluating the untrained model ===")
        base_model = load_fresh_model(args)
        if torch.cuda.is_available():
            base_model = base_model.cuda()
        base_metrics = evaluate(base_model, tokenizer, eval_ds, args)
        pd.DataFrame(base_metrics.pop("predictions")).to_json(
            out_root / "base_predictions.jsonl", orient="records", lines=True
        )
        base_path.write_text(json.dumps(base_metrics, indent=2))
        print(f"untrained: accuracy={base_metrics['accuracy']:.4f}")
        del base_model
        cleanup()
    base_accuracy = json.loads(base_path.read_text())["accuracy"] if base_path.exists() else float("nan")

    sft_ids = set(json.loads(Path(args.sft_ids).read_text())) if args.sft_ids else None
    for method in methods:
        print(f"\n{'='*80}\nEXPERIMENT: {method}\n{'='*80}")
        # Reset RNG so a method does not inherit random state consumed by earlier methods.
        set_seed(args.seed)
        exp_dir = out_root / method
        exp_dir.mkdir(parents=True, exist_ok=True)

        sft_ds, grpo_ds = method_partition(method, train_ds, router_df, args.seed, sft_ids)
        if args.grpo_on_sft_prompts:  # RL reuses the SFT problems instead of the held-out bucket
            grpo_ds = sft_ds
        parsed = method_metadata(method, len(train_ds))
        strategy = str(parsed["strategy"])
        if method == "adaptive":
            strategy = "adaptive"
        n_sft, n_grpo = len(sft_ds), len(grpo_ds)
        sft_fraction = n_sft / max(len(train_ds), 1)
        sft_ds = with_sft_targets(sft_ds, args.sft_traces)

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
        after_sft_accuracy = float("nan")

        # Every mixed experiment uses exactly this order:
        # base -> SFT on SFT bucket -> GRPO on disjoint GRPO bucket (or on the SFT problems, --grpo_on_sft_prompts).
        # Adapters are saved to <method>/sft and <method>/grpo after each stage.
        if len(sft_ds) and args.reuse_from:
            adapter = Path(args.reuse_from) / method / args.reuse_stage
            print(f"[stage 1] adapter reused from {adapter}")
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
            if torch.cuda.is_available():
                model = model.cuda()
        elif len(sft_ds):
            print("[stage 1] SFT")
            if wandb_run is not None:
                wandb_run.log({"stage/sft_started": 1})
            model = train_sft(model, tokenizer, sft_ds, args, exp_dir / "sft")

        if len(sft_ds) and len(grpo_ds) and not args.skip_grpo:
            print("[eval] after SFT, before GRPO")
            sft_metrics = evaluate(model, tokenizer, eval_ds, args)
            pd.DataFrame(sft_metrics.pop("predictions")).to_json(
                exp_dir / "predictions_after_sft.jsonl", orient="records", lines=True
            )
            after_sft_accuracy = sft_metrics["accuracy"]

        if len(grpo_ds) and not args.skip_grpo:
            print("[stage 2] GRPO")
            if wandb_run is not None:
                wandb_run.log({"stage/grpo_started": 1})
            model = train_grpo(model, tokenizer, grpo_ds, args, exp_dir / "grpo",
                               panel_prompts=eval_ds["prompt"][: args.panel_n])

        print("[eval] greedy exact-match")
        metrics = evaluate(model, tokenizer, eval_ds, args)
        predictions = metrics.pop("predictions")
        pd.DataFrame(predictions).to_json(
            exp_dir / "predictions.jsonl", orient="records", lines=True
        )
        if len(sft_ds) and (not len(grpo_ds) or args.skip_grpo):  # the final model is the SFT model
            after_sft_accuracy = metrics["accuracy"]

        # GRPO starts from the SFT model if there was SFT, otherwise from the untrained model.
        grpo_start = after_sft_accuracy if len(sft_ds) else base_accuracy

        row = {
            "method": method,
            "strategy": strategy,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "n_total": len(train_ds),
            "n_sft": n_sft,
            "n_grpo": n_grpo,
            "sft_fraction": sft_fraction,
            "sft_traces": args.sft_traces,
            "reward": args.reward,
            "n_sft_trained": len(sft_ds),
            **metrics,
            "base_accuracy": base_accuracy,
            "after_sft_accuracy": after_sft_accuracy,
            "sft_gain": after_sft_accuracy - base_accuracy,  # NaN when there is no SFT
            "grpo_gain": metrics["accuracy"] - grpo_start if len(grpo_ds) else float("nan"),
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
    p.add_argument("--dataset", choices=["gsm8k", "math", "geometry3k"], default="gsm8k",
                   help="geometry3k: diagram questions for a VLM (e.g. Qwen/Qwen2.5-VL-3B-Instruct), use --prompt_style boxed.")
    p.add_argument("--prompt_style", choices=list(PROMPT_STYLES), default="train", help="Prompt used for training and eval.")
    p.add_argument("--output_dir", default="./sft_grpo_routing_runs")
    p.add_argument("--seed", type=int, default=42, help="Training / partition seed.")
    p.add_argument("--grpo_seed", type=int, default=None, help="Seed of the GRPO stage only (default: --seed).")
    p.add_argument(
        "--reuse_from",
        "--reuse_sft_from",
        dest="reuse_from",
        default="",
        help="Seed dir of an earlier run: start from its <method>/<reuse_stage> adapter instead of training SFT.",
    )
    p.add_argument("--skip_grpo", action="store_true", help="Stop after SFT (SFT-only dose/LR scans).")
    p.add_argument("--grpo_on_sft_prompts", action="store_true",
                   help="Run GRPO on the SFT problems rather than the disjoint remainder (e.g. full_sft then GRPO on all).")
    p.add_argument(
        "--reuse_stage",
        choices=["sft", "grpo"],
        default="sft",
        help="grpo = continue from the earlier run's final adapter (one more GRPO epoch).",
    )
    p.add_argument(
        "--data_seed",
        type=int,
        default=1234,
        help="Fixed seed for selecting the dataset subset. Keep constant across training seeds.",
    )
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument(
        "--sft_traces",
        default="gold",
        help="SFT targets: gold | stripped (gold without <<calc>>) | path to a probe.py .jsonl (first correct completion). "
             "geometry3k has no gold solutions ('gold' = the boxed answer alone).",
    )
    p.add_argument("--sft_ids", default=None, help="JSON list of example_ids for method listed_sft (the rest go to GRPO).")

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
    p.add_argument("--grpo_steps", type=int, default=-1, help="Fixed number of GRPO steps (cycles the bucket); -1 = use --grpo_epochs.")
    p.add_argument("--grpo_lr_scheduler", default="linear", help="e.g. linear (decay to 0) or constant_with_warmup.")
    p.add_argument("--grpo_warmup_steps", type=int, default=0)
    p.add_argument("--grpo_save_steps", type=int, default=0, help="Checkpoint every this many GRPO steps (0 = only the final adapter); a requeued run resumes from the last one.")
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
    p.add_argument("--reward", choices=["lenient", "marked"], default="lenient",
                   help="lenient: the last number counts as the answer; marked: only an explicitly marked answer "
                        "(#### N, \\boxed{N}, 'answer is N'; geometry3k: \\boxed{} or ####). "
                        "Evaluation is graded the same way and also reports the other.")
    p.add_argument("--reward_needs_stop", action="store_true",
                   help="Reward an answer only if the completion ended before the length limit (no reward for "
                        "repeating '#### N' until truncation).")
    p.add_argument("--censor", default=None,
                   help="Regex: GRPO rollouts that match get no reward and no advantage, so that behaviour is never "
                        "reinforced (e.g. '## Step'). Measure it with --panel_prefix.")
    p.add_argument("--log_rollouts", action="store_true", help="Write every GRPO rollout to <method>/grpo/rollouts.jsonl.")
    p.add_argument("--panel_prefix", nargs="*", default=[],
                   help="Teacher-forced log p(completion starts with PREFIX) on the first --panel_n test prompts, "
                        "at GRPO step 0 and every --panel_every steps (<method>/grpo/panel.jsonl).")
    p.add_argument("--panel_n", type=int, default=256)
    p.add_argument("--panel_every", type=int, default=10)

    # Evaluation.
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--eval_max_new_tokens", type=int, default=512)
    p.add_argument("--eval_temperature", type=float, default=0.0, help="0 = greedy; > 0 samples --eval_k answers per question.")
    p.add_argument("--eval_k", type=int, default=1)

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
        default="sft-grpo-routing",
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

    if args.grpo_seed is None:
        args.grpo_seed = args.seed
    run(args)
