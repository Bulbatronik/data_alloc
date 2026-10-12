#!/usr/bin/env python3
"""
Classify completions into answer styles ("modes") with surface rules, and summarize them.

    gold    reference-trace style: GSM8K calculator annotations like <<2*4=8>>, or MATH solutions
            (\\boxed{} followed by the '#### <answer>' line we append to MATH gold traces)
    native  the model's own math style: "## Step" or \\boxed{}
            (split_native=True splits it into "step" and "boxed", where "## Step" wins)
    prompt  follows our system prompt: ends with '#### <answer>'
    other   none of the above

Rules are checked in this order, so a completion with <<...>> is gold even if it also ends in '####'.

Usage (any predictions.jsonl / probe.py output):
    python modes.py 'runs/gsm8k/Llama-3.2-1B-Instruct/seed_*/*/predictions.jsonl'
    python modes.py 'runs/probes/t1/*.jsonl' --by model dataset prompt_style
"""

from __future__ import annotations

import argparse
import glob
import re

import pandas as pd

MODES = ["gold", "native", "prompt", "other"]
_GOLD = re.compile(r"<<[^<>]*=[^<>]*>>")
_NATIVE = re.compile(r"## Step|\\boxed\{")


def classify(text: str, split_native: bool = False) -> str:
    if _GOLD.search(text) or ("\\boxed{" in text and "####" in text):
        return "gold"
    if _NATIVE.search(text):
        if split_native:
            return "step" if "## Step" in text else "boxed"
        return "native"
    if "####" in text:
        return "prompt"
    return "other"


def mode_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group: share of each mode (w_), accuracy within each mode (p_), and overall accuracy."""
    df = df.assign(mode=df["completion"].map(classify))
    share = pd.crosstab([df[c] for c in by], df["mode"], normalize="index").reindex(columns=MODES, fill_value=0)
    acc = df.pivot_table(index=by, columns="mode", values="correct", aggfunc="mean").reindex(columns=MODES)
    out = share.add_prefix("w_").join(acc.add_prefix("p_"))
    out["n"] = df.groupby(by).size()
    out["accuracy"] = df.groupby(by)["correct"].mean()
    return out


def load(patterns: list[str]) -> pd.DataFrame:
    files = sorted(f for p in patterns for f in glob.glob(p))
    if not files:
        raise SystemExit(f"No files match {patterns}")
    return pd.concat((pd.read_json(f, lines=True).assign(file=f) for f in files), ignore_index=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("patterns", nargs="+", help="Globs of .jsonl files with `completion` and `correct` columns.")
    p.add_argument("--by", nargs="+", default=["file"], help="Columns to group by (default: file).")
    args = p.parse_args()
    pd.set_option("display.width", 250)
    print(mode_table(load(args.patterns), args.by).round(3).to_string())
