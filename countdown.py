#!/usr/bin/env python3
"""
Countdown: combine the given numbers with + - * / to reach a target (TinyZero; Gandhi et al. 2503.01307).

Convention: every number is used EXACTLY once. TinyZero's prompt says "each number can only be used once", but its
verifier (github.com/Jiayi-Pan/TinyZero, verl/utils/reward_score/countdown.py, validate_equation: sorted numbers in
the equation == sorted available numbers; opened 2026-10-19) requires all of them, once each; we follow the
verifier and say "exactly once" in the question. TinyZero's data (Jiayi-Pan/Countdown-Tasks-3to4: 3-4 numbers,
targets 10-100 per the HF dataset statistics) comes from a generator that draws the target and the numbers
independently (examples/data_preprocess/countdown.py, gen_dataset). Ours draws n numbers uniformly in [1, 100], then
the target uniformly among the integers in [target_min, target_max] that the numbers reach, so every item is solvable
by construction (and re-checked by the solver).

Seeds: item i of a split is drawn with random.Random(SEEDS[split] + i); the splits use disjoint seed ranges, and an
instance (multiset of numbers, target) already drawn for an earlier split (test first) is redrawn.

Verifier (format-insensitive): the candidate is the last <answer>...</answer>, else the last \\boxed{...}, else the
last line that contains an arithmetic expression. Inside it the arithmetic span is evaluated exactly (fractions; a
small recursive-descent parser, no eval). The candidate may be an equation ("(44 + 19) + 35 = 98"): it is correct
if every '='-separated side evaluates to the target and one side uses exactly the given multiset of numbers.
marked=True accepts only an <answer>...</answer> tag.

    python countdown.py --out data/countdown            # writes train.jsonl (sft + rl pools), test.jsonl, meta.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Disjoint seed ranges per split (1e6 items each at most).
SEEDS = {"test": 3_000_000, "sft": 1_000_000, "rl": 2_000_000}
NUM_RANGE = (1, 100)

QUESTION = ("Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations "
            "(+, -, *, /) and each number must be used exactly once.")


def question(nums: Sequence[int], target: int) -> str:
    return QUESTION.format(nums=list(nums), target=target)


def gold_answer(nums: Sequence[int], target: int) -> str:
    """The `gold_answer` string the pipeline passes to the verifier."""
    return json.dumps({"target": int(target), "nums": [int(n) for n in nums]})


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _results(a: Fraction, b: Fraction):
    yield "+", a + b
    yield "*", a * b
    yield "-", a - b
    yield "-r", b - a
    if b != 0:
        yield "/", a / b
    if a != 0:
        yield "/r", b / a


@lru_cache(maxsize=200_000)
def reachable(nums: Tuple[Fraction, ...]) -> frozenset:
    """Every value an expression using each of `nums` exactly once can take (nums sorted)."""
    if len(nums) == 1:
        return frozenset(nums)
    out = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            rest = nums[:i] + nums[i + 1:j] + nums[j + 1:]
            for _, v in _results(nums[i], nums[j]):
                out |= reachable(tuple(sorted(rest + (v,))))
    return frozenset(out)


def _fmt(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


# precedence: 3 atom, 2 * /, 1 + -
def _wrap(expr: str, prec: int, need: int) -> str:
    return f"({expr})" if prec < need else expr


def _combine(a, b, op):
    """(value, expr, prec) of a op b for a natural step: op in + * - / with a >= b for - and /."""
    (va, ea, pa), (vb, eb, pb) = a, b
    if op == "+":
        return va + vb, f"{ea} + {eb}", 1
    if op == "*":
        return va * vb, f"{_wrap(ea, pa, 2)} * {_wrap(eb, pb, 2)}", 2
    if op == "-":
        return va - vb, f"{ea} - {_wrap(eb, pb, 2)}", 1
    return va / vb, f"{_wrap(ea, pa, 2)} / {_wrap(eb, pb, 3)}", 2


def solve(nums: Sequence[int], target: int):
    """Depth-first search for an expression using each number exactly once. Returns (steps, expr) or None; a step is
    (left value, op, right value, result, numbers left). Natural steps first (non-negative, integer intermediates;
    larger operand first for - and /), then any rational intermediates."""
    target = Fraction(target)

    def dfs(items, steps, natural):
        if len(items) == 1:
            return (steps, items[0][1]) if items[0][0] == target else None
        if target not in reachable(tuple(sorted(v for v, _, _ in items))):
            return None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a[0] < b[0]:
                    a, b = b, a
                rest = [x for k, x in enumerate(items) if k not in (i, j)]
                for op in "+*-/":
                    pairs = [(a, b)] if natural or op in "+*" else [(a, b), (b, a)]
                    for x, y in pairs:
                        if op == "/" and y[0] == 0:
                            continue
                        v, e, p = _combine(x, y, op)
                        if natural and (v < 0 or v.denominator != 1):
                            continue
                        left = rest + [(v, e, p)]
                        found = dfs(left, steps + [(x[0], op, y[0], v, [w for w, _, _ in left])], natural)
                        if found:
                            return found
        return None

    start = [(Fraction(n), str(n), 3) for n in nums]
    return dfs(start, [], True) or dfs(start, [], False)


def solver_trace(nums: Sequence[int], target: int) -> Tuple[str, str]:
    """(natural-language step-by-step solution ending in <answer>expr</answer>, expr)."""
    found = solve(nums, target)
    if found is None:
        raise ValueError(f"unsolvable: {nums} -> {target}")
    steps, expr = found
    lines = [f"We need to reach {target} using {', '.join(map(str, nums))}, each exactly once."]
    for k, (a, op, b, v, left) in enumerate(steps, 1):
        tail = (f"Numbers left: {', '.join(_fmt(w) for w in left)}." if len(left) > 1
                else "This is the target.")
        lines.append(f"Step {k}: {_fmt(a)} {op} {_fmt(b)} = {_fmt(v)}. {tail}")
    lines.append(f"So the equation is {expr} = {target}.")
    lines.append(f"<answer>{expr}</answer>")
    return "\n".join(lines), expr


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def draw(rng: random.Random, n_numbers: int, target_min: int, target_max: int):
    """Numbers uniform in NUM_RANGE, then a target uniform among the reachable integers in [target_min, target_max]."""
    while True:
        nums = [rng.randint(*NUM_RANGE) for _ in range(n_numbers)]
        values = reachable(tuple(sorted(Fraction(n) for n in nums)))
        targets = sorted(int(v) for v in values if v.denominator == 1 and target_min <= v <= target_max)
        if targets:
            return nums, rng.choice(targets), len(targets)


def make_split(split: str, n: int, n_numbers: Sequence[int], target_min: int, target_max: int, seen: set):
    rows = []
    for i in range(n):
        seed = SEEDS[split] + i
        rng = random.Random(seed)
        k = n_numbers[i % len(n_numbers)]  # balanced mix of 3- and 4-number items
        while True:
            nums, target, n_targets = draw(rng, k, target_min, target_max)
            key = (tuple(sorted(nums)), target)
            if key not in seen:
                break
        seen.add(key)
        trace, expr = solver_trace(nums, target)
        assert is_correct(expr, nums, target) and is_correct(trace, nums, target, marked=True), (nums, target, expr)
        rows.append({"id": f"cd-{split}-{i:06d}", "pool": split, "seed": seed, "nums": nums, "target": target,
                     "n_numbers": k, "n_reachable_targets": n_targets, "solution": expr, "trace": trace})
    reachable.cache_clear()
    return rows


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_REWRITES = [
    (r"\\(?:times|cdot|ast)|[×·∗⋅]", "*"),
    (r"\\div|÷|∕", "/"),
    (r"[−–—]", "-"),
    (r"\\left|\\right|\\[,;!: ]|\$|\\\(|\\\)|\\\[|\\\]", " "),
    (r"\*\*", " "),  # markdown bold (there is no exponent in Countdown)
    (r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*"),  # 3 x 4
]


def _last_boxed(text: str) -> Optional[str]:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, begin = 0, start + len("\\boxed{")
    for i in range(begin, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[begin:i]
            depth -= 1
    return None


def _normalize(s: str) -> str:
    for pattern, repl in _REWRITES:
        s = re.sub(pattern, repl, s)
    while True:  # \frac{a}{b} -> (a)/(b), innermost first
        t = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
        if t == s:
            return s
        s = t


def arithmetic_span(s: str) -> Optional[str]:
    """The last run of digits, + - * / ( ) . = and spaces in s with >= 2 numbers and an operator."""
    s = re.sub(r"^\s*\d+[.)]\s+", "", _normalize(s))  # a list enumerator "2. "
    best = None
    for run in re.findall(r"[\d\s+\-*/().=]+", s):
        for piece in re.split(r"\.(?=\s|$)", run):  # a sentence-ending period
            piece = piece.strip().strip("=").strip()
            if len(re.findall(r"\d+", piece)) >= 2 and re.search(r"[+\-*/]", piece):
                best = piece
    return best


def extract_equation(text: str, marked: bool = False) -> Optional[str]:
    tags = _TAG.findall(text)
    if tags:
        return tags[-1].strip()
    if marked:
        return None
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    for line in reversed(text.splitlines()):
        span = arithmetic_span(line)
        if span:
            return span
    return None


def _tokens(s: str) -> Optional[List[str]]:
    toks = re.findall(r"\d+(?:\.\d+)?|[+\-*/()]|\S", s)
    return toks if all(re.fullmatch(r"\d+(?:\.\d+)?|[+\-*/()]", t) for t in toks) else None


def evaluate(s: str) -> Optional[Tuple[Fraction, List[Fraction]]]:
    """(exact value, number literals) of an arithmetic expression with + - * / ( ) and unary signs; None if invalid
    or dividing by zero."""
    toks = _tokens(s)
    if not toks or len(toks) > 200:
        return None
    pos, numbers = 0, []

    def peek():
        return toks[pos] if pos < len(toks) else None

    def expr(depth):
        nonlocal pos
        v = term(depth)
        while peek() in ("+", "-"):
            op = toks[pos]
            pos += 1
            w = term(depth)
            v = v + w if op == "+" else v - w
        return v

    def term(depth):
        nonlocal pos
        v = factor(depth)
        while peek() in ("*", "/"):
            op = toks[pos]
            pos += 1
            w = factor(depth)
            if op == "/" and w == 0:
                raise ZeroDivisionError
            v = v * w if op == "*" else v / w
        return v

    def factor(depth):
        nonlocal pos
        if depth > 50:
            raise ValueError("too deep")
        t = peek()
        if t in ("+", "-"):
            pos += 1
            v = factor(depth + 1)
            return v if t == "+" else -v
        if t == "(":
            pos += 1
            v = expr(depth + 1)
            if peek() != ")":
                raise ValueError("unbalanced")
            pos += 1
            return v
        if t is not None and t[0].isdigit():
            pos += 1
            v = Fraction(t)
            numbers.append(v)
            return v
        raise ValueError(f"unexpected {t!r}")

    try:
        value = expr(0)
    except (ValueError, ZeroDivisionError):
        return None
    return (value, numbers) if pos == len(toks) else None


def is_correct(candidate: Optional[str], nums: Sequence[int], target: int, marked: bool = False) -> bool:
    """candidate: an extracted answer (or a whole completion, which is extracted first if it has a tag/box/line)."""
    if candidate is None:
        return False
    if "<answer>" in candidate.lower() or "\\boxed{" in candidate or "\n" in candidate:
        candidate = extract_equation(candidate, marked)
        if candidate is None:
            return False
    span = arithmetic_span(candidate)
    if span is None:
        return False
    sides = [evaluate(side) for side in span.split("=")]
    if any(s is None for s in sides) or any(v != target for v, _ in sides):
        return False
    want = sorted(Fraction(n) for n in nums)
    return any(sorted(used) == want for _, used in sides)


def answer_is_correct(completion: str, gold: str, marked: bool = False) -> bool:
    g = json.loads(gold)
    return is_correct(extract_equation(completion, marked), g["nums"], g["target"])


# ---------------------------------------------------------------------------
# CLI: write the fixed splits
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/countdown")
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_sft", type=int, default=4000, help="SFT pool (train.jsonl, pool=sft).")
    p.add_argument("--n_rl", type=int, default=4000, help="RL pool (train.jsonl, pool=rl).")
    p.add_argument("--n_numbers", default="3,4", help="Numbers per item, cycled (balanced mix).")
    p.add_argument("--target_min", type=int, default=1)
    p.add_argument("--target_max", type=int, default=999)
    args = p.parse_args()

    n_numbers = [int(x) for x in args.n_numbers.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    splits = {}
    for split, n in (("test", args.n_test), ("sft", args.n_sft), ("rl", args.n_rl)):
        splits[split] = make_split(split, n, n_numbers, args.target_min, args.target_max, seen)
        print(f"{split}: {len(splits[split])} items", flush=True)
    for name, rows in (("test", splits["test"]), ("train", splits["sft"] + splits["rl"])):
        with open(out / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    meta = {"seeds": SEEDS, "num_range": NUM_RANGE, "n_numbers": n_numbers, "target_range": [args.target_min, args.target_max],
            "counts": {k: len(v) for k, v in splits.items()}, "convention": "each number exactly once (TinyZero verifier)",
            "generator": "numbers uniform in num_range, target uniform among reachable integers in target_range"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
