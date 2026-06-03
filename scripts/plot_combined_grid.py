"""
Plot all rubric-hacking runs in a 2×4 tiled grid.

Usage:
    conda run -n rubrics_wxk python scripts/plot_combined_grid.py \
        --output figures/combined_grid.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import wandb

ENTITY = "wxk20040223-tsinghua-university"

RUNS = [
    # (title,             project,                   run_id,      vline_x)
    ("Verif / Self-Praise",  "verl_grpo_rubrics_verif",  "si4wyh62", 478),
    ("Verif / Lexical",      "verl_grpo_rubrics_verif",  "8fg0zx9j", 116),
    ("Verif / Format",       "verl_grpo_rubrics_verif",  "n0vxrls5", 301),
    ("Health / Self-Praise", "verl_grpo_healthbench",    "2kctoqeu", 460),
    ("Health / Lexical",     "verl_grpo_healthbench",    "4w9nfhat", 91),
    ("Health / Tone",        "verl_grpo_healthbench",    "v3rd7gwl", 68),
    ("Verif / Tone",         "verl_grpo_rubrics_verif",  "6a57to01", None),  # G
    ("Health / Format",      "verl_grpo_healthbench",    "2e1jyasn", None),  # H
]

METRICS = ["reward_extra/acc/mean", "critic/rewards/mean"]
LABELS  = ["Gold Reward", "Proxy Reward"]
COLORS  = ["#589C3F", "#4677AF"]
ALPHA   = 0.9


def fetch(entity, project, run_id, metrics, timeout=120, max_retries=3):
    api = wandb.Api(timeout=timeout)
    run = api.run(f"{entity}/{project}/{run_id}")
    total = run.lastHistoryStep + 1
    for attempt in range(1, max_retries + 1):
        try:
            df = run.history(samples=total + 100, keys=["_step"] + metrics, pandas=True)
            df = df.sort_values("_step").reset_index(drop=True)
            return df
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  Attempt {attempt} failed ({e}), retrying...")


def smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1, center=True).mean()


def plot_one(ax, df, title, smooth_window, alpha=ALPHA, vline_x=None):
    for metric, label, color in zip(METRICS, LABELS, COLORS):
        if metric not in df.columns:
            print(f"  Warning: '{metric}' not found in {title}")
            continue
        x = df["_step"]
        y = smooth(df[metric], smooth_window)
        ax.plot(x, y, linewidth=1.5, label=label, color=color, alpha=alpha)
        if smooth_window > 1:
            ax.plot(x, df[metric], alpha=min(alpha * 0.17, 0.3), linewidth=0.6, color=color)

    if vline_x is not None:
        ax.axvline(x=vline_x, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_title("")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")


SUBPLOT_LABELS = [chr(ord('A') + i) for i in range(26)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth",     type=int,   default=3)
    parser.add_argument("--output",     default="figures/combined_grid.svg")
    parser.add_argument("--title",      default="Training Dynamics of Reward Hacking in Rubrics RL")
    parser.add_argument("--figsize",    nargs=2, type=float, default=[20, 9])
    parser.add_argument("--alpha",      type=float, default=ALPHA,
                        help="line opacity for smoothed curves (0.0–1.0, default %(default)s)")
    parser.add_argument("--individual", action="store_true",
                        help="save each subplot as a separate file named A.png, B.png, ...")
    parser.add_argument("--sub-figsize", nargs=2, type=float, default=[5, 3],
                        dest="sub_figsize",
                        help="figsize for each individual subplot (default: 5 4)")
    args = parser.parse_args()

    outdir = os.path.dirname(args.output) or "."
    os.makedirs(outdir, exist_ok=True)

    if args.individual:
        for idx, (title, project, run_id, vline_x) in enumerate(RUNS):
            label = SUBPLOT_LABELS[idx]
            print(f"[{idx+1}/{len(RUNS)}] Fetching {title} ({run_id}) ...")
            df = fetch(ENTITY, project, run_id, METRICS)
            print(f"  {len(df)} steps")
            fig, ax = plt.subplots(figsize=args.sub_figsize)
            plot_one(ax, df, title, args.smooth, alpha=args.alpha, vline_x=vline_x)
            fig.tight_layout()
            out_path = os.path.join(outdir, f"{label}.svg")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {out_path}")
        return

    # 2-row × 4-col grid; row 0: verif × 4, row 1: health × 4
    NCOLS, NROWS = 4, 2
    fig = plt.figure(figsize=args.figsize)
    gs  = gridspec.GridSpec(NROWS, NCOLS, figure=fig, hspace=0.45, wspace=0.35)

    positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
    ]

    for idx, ((row, col), (title, project, run_id, vline_x)) in enumerate(zip(positions, RUNS)):
        print(f"[{idx+1}/{len(RUNS)}] Fetching {title} ({run_id}) ...")
        df = fetch(ENTITY, project, run_id, METRICS)
        print(f"  {len(df)} steps")
        ax = fig.add_subplot(gs[row, col])
        plot_one(ax, df, title, args.smooth, alpha=args.alpha, vline_x=vline_x)

    fig.suptitle(args.title, fontsize=14, fontweight="bold", y=1.01)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
