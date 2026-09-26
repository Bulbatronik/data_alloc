#!/usr/bin/env python3
"""
Run an SFT/GRPO allocation curve and log it to Weights & Biases.

The x-axis is the exact number of SFT examples. The GRPO bucket always contains
N - num_sft examples. For intermediate allocations, both random and adaptive
routing can be run. The endpoints are shared:

    num_sft = 0  -> full GRPO
    num_sft = N  -> full SFT

Example (the five originally requested points, one seed):

    python run_curve.py \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --max_train_samples 1000 \
      --max_eval_samples 300 \
      --fractions 0 0.25 0.50 0.75 1.0 \
      --seeds 42 \
      --bf16

A denser curve with three seeds:

    python run_curve.py \
      --max_train_samples 1000 \
      --fractions 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
      --seeds 42 43 44 \
      --bf16

Unknown arguments are forwarded to route_sft_grpo.py, so flags such as --bf16,
--gradient_checkpointing, --sft_epochs, --grpo_epochs, etc. work directly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max_train_samples", type=int, default=1000)
    p.add_argument("--max_eval_samples", type=int, default=300)
    p.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="SFT fractions. Ignored when --counts is provided.",
    )
    p.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=None,
        help="Exact num_sft x-axis values. Endpoints 0 and N are added automatically.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Training / partition seeds. Use >=3 for error bars.",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        choices=["random", "adaptive"],
        default=["random", "adaptive"],
    )
    p.add_argument("--data_seed", type=int, default=1234)
    p.add_argument("--output_dir", default="", help="Defaults to runs/<model name>.")
    p.add_argument("--wandb_project", default="sft-grpo-routing")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_group", default="", help="Defaults to the model name.")
    p.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    p.add_argument(
        "--include_natural_adaptive",
        action="store_true",
        help="Also run the router's unconstrained natural SFT/GRPO split.",
    )
    p.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a seed if its results.csv already contains all requested methods.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without launching training.",
    )
    p.add_argument(
        "--no_aggregate",
        action="store_true",
        help="Only train (used by per-seed Slurm jobs); aggregate later in one call.",
    )
    return p.parse_known_args()


def get_n_total(max_train_samples: int) -> int:
    if max_train_samples > 0:
        return max_train_samples
    # Full GSM8K requested; ask the dataset instead of hard-coding its size.
    from datasets import load_dataset

    return len(load_dataset("openai/gsm8k", "main", split="train"))


def counts_from_fractions(fractions: Iterable[float], n_total: int) -> List[int]:
    counts = []
    for f in fractions:
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"SFT fraction {f} is outside [0, 1].")
        counts.append(round(f * n_total))
    counts.extend([0, n_total])
    return sorted(set(counts))


def build_methods(counts: List[int], n_total: int, strategies: List[str], include_natural: bool):
    methods = []
    for n_sft in counts:
        if n_sft == 0:
            methods.append("full_grpo")
        elif n_sft == n_total:
            methods.append("full_sft")
        else:
            for strategy in strategies:
                methods.append(f"{strategy}_sft_{n_sft}")
    if include_natural and "adaptive" in strategies:
        methods.append("adaptive")
    return methods


def already_complete(results_path: Path, methods: List[str]) -> bool:
    if not results_path.exists():
        return False
    try:
        found = set(pd.read_csv(results_path)["method"].astype(str))
    except Exception:
        return False
    return set(methods).issubset(found)


def infer_strategy(method: str) -> str:
    if method.startswith("random"):
        return "random"
    if method.startswith("adaptive"):
        return "adaptive"
    if method in {"full_sft", "full_grpo"}:
        return "endpoint"
    return "other"


def expand_endpoints_for_curves(raw: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    """Duplicate shared endpoint rows so every strategy's line reaches x=0 and x=N."""
    pieces = [raw[raw["strategy"] != "endpoint"].copy()]
    endpoints = raw[raw["strategy"] == "endpoint"].copy()
    for strategy in strategies:
        clone = endpoints.copy()
        clone["strategy"] = strategy
        pieces.append(clone)
    return pd.concat(pieces, ignore_index=True)


def aggregate_curve(raw_for_plot: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        raw_for_plot.groupby(["strategy", "n_total", "n_sft", "n_grpo", "sft_fraction"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            n_seeds=("accuracy", "count"),
        )
        .sort_values(["strategy", "n_sft"])
    )
    grouped["accuracy_std"] = grouped["accuracy_std"].fillna(0.0)
    return grouped


def make_local_plot(summary: pd.DataFrame, n_total: int, out_png: Path, out_pdf: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for strategy, df in summary.groupby("strategy"):
        df = df.sort_values("n_sft")
        yerr = df["accuracy_std"] if (df["n_seeds"] > 1).any() else None
        ax.errorbar(
            df["n_sft"],
            df["accuracy_mean"],
            yerr=yerr,
            marker="o",
            capsize=3,
            label=strategy,
        )

    ax.set_xlabel("Number of SFT examples")
    ax.set_ylabel("Final GSM8K exact-match accuracy")
    ax.set_title("SFT / GRPO allocation curve")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Allocation rule")

    # Since num_grpo = N - num_sft, show that budget explicitly on top.
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda x: n_total - x, lambda x: n_total - x),
    )
    sec.set_xlabel("Number of GRPO examples (= N - SFT)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def log_summary_to_wandb(args, group_name: str, raw: pd.DataFrame, summary: pd.DataFrame, plot_png: Path):
    if args.wandb_mode == "disabled":
        return

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=group_name,
        name=f"curve-summary-{group_name}",
        job_type="curve-summary",
        mode=args.wandb_mode,
        config={
            "model": args.model,
            "n_total": int(raw["n_total"].iloc[0]),
            "seeds": args.seeds,
            "fractions": args.fractions if args.counts is None else None,
            "counts": args.counts,
            "strategies": args.strategies,
            "data_seed": args.data_seed,
        },
    )

    raw_cols = [
        "strategy", "seed", "n_total", "n_sft", "n_grpo", "sft_fraction", "accuracy"
    ]
    raw_table = wandb.Table(
        columns=raw_cols,
        data=raw[raw_cols].sort_values(["strategy", "n_sft", "seed"]).values.tolist(),
    )

    summary_cols = [
        "strategy", "n_total", "n_sft", "n_grpo", "sft_fraction",
        "accuracy_mean", "accuracy_std", "n_seeds",
    ]
    summary_table = wandb.Table(
        columns=summary_cols,
        data=summary[summary_cols].values.tolist(),
    )

    line_plot = wandb.plot.line(
        table=summary_table,
        x="n_sft",
        y="accuracy_mean",
        stroke="strategy",
        title="Final performance vs number of SFT examples",
    )

    run.log(
        {
            "allocation/raw_runs": raw_table,
            "allocation/curve_table": summary_table,
            "allocation/final_accuracy_vs_num_sft": line_plot,
            "allocation/local_curve": wandb.Image(str(plot_png)),
        }
    )
    run.finish()


def main():
    args, passthrough = parse_args()
    n_total = get_n_total(args.max_train_samples)
    if args.counts is not None:
        bad = [c for c in args.counts if c < 0 or c > n_total]
        if bad:
            raise ValueError(f"SFT counts outside [0, {n_total}]: {bad}")
        counts = sorted(set([0, n_total, *args.counts]))
    else:
        counts = counts_from_fractions(args.fractions, n_total)
    methods = build_methods(
        counts,
        n_total=n_total,
        strategies=args.strategies,
        include_natural=args.include_natural_adaptive,
    )

    model_name = args.model.split("/")[-1]
    out_root = Path(args.output_dir or f"runs/{model_name}")
    out_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("route_sft_grpo.py")

    group_name = args.wandb_group or model_name

    plan = {
        "model": args.model,
        "n_total": n_total,
        "counts": counts,
        "fractions_requested": args.fractions if args.counts is None else None,
        "counts_requested": args.counts,
        "strategies": args.strategies,
        "seeds": args.seeds,
        "methods": methods,
        "wandb_group": group_name,
    }
    (out_root / "sweep_plan.json").write_text(json.dumps(plan, indent=2))

    print("\nAllocation curve plan")
    print(json.dumps(plan, indent=2))

    for seed in args.seeds:
        seed_dir = out_root / f"seed_{seed}"
        results_path = seed_dir / "results.csv"

        if args.skip_existing and already_complete(results_path, methods):
            print(f"[seed {seed}] all requested methods already present; skipping.")
            continue

        cmd = [
            sys.executable,
            str(script),
            "--model", args.model,
            "--max_train_samples", str(args.max_train_samples),
            "--max_eval_samples", str(args.max_eval_samples),
            "--seed", str(seed),
            "--data_seed", str(args.data_seed),
            "--output_dir", str(seed_dir),
            "--wandb_project", args.wandb_project,
            "--wandb_group", group_name,
            "--wandb_mode", args.wandb_mode,
            "--methods", *methods,
        ]
        if args.wandb_entity:
            cmd.extend(["--wandb_entity", args.wandb_entity])
        cmd.extend(passthrough)

        print("\n[launch]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if args.dry_run or args.no_aggregate:
        return

    frames = []
    for seed in args.seeds:
        result_path = out_root / f"seed_{seed}" / "results.csv"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing {result_path}")
        df = pd.read_csv(result_path)
        df = df[df["method"].isin(methods)].copy()
        if "strategy" not in df.columns:
            df["strategy"] = df["method"].map(infer_strategy)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw_path = out_root / "curve_raw.csv"
    raw.to_csv(raw_path, index=False)

    raw_for_plot = expand_endpoints_for_curves(raw, args.strategies)
    # If --include_natural_adaptive is used, its unconstrained allocation is
    # included as an extra adaptive point in addition to the budget grid.
    summary = aggregate_curve(raw_for_plot)
    summary_path = out_root / "curve_summary.csv"
    summary.to_csv(summary_path, index=False)

    png = out_root / "allocation_curve.png"
    pdf = out_root / "allocation_curve.pdf"
    make_local_plot(summary, n_total, png, pdf)

    log_summary_to_wandb(args, group_name, raw_for_plot, summary, png)

    print("\nDone.")
    print(f"Raw runs:      {raw_path}")
    print(f"Curve summary: {summary_path}")
    print(f"PNG curve:     {png}")
    print(f"PDF curve:     {pdf}")
    print(f"W&B group:     {group_name}")


if __name__ == "__main__":
    main()
