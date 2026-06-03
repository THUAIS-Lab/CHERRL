#!/usr/bin/env python3
"""Plot a budget-vs-onset summary figure from a budget-ablation summary CSV.

Reads the per-budget aggregate produced by ``parse.py`` (or any compatible
CSV with columns ``budget`` and a numeric onset-mean column), and writes a
clean line plot of mean onset across budgets.

Usage
-----
    export RUN_ID=run_e   # or any run id with a per-budget summary CSV

    python -m detection.agent_compare.budget_ablation.plot \\
        --run "$RUN_ID" \\
        --input-csv /path/to/restored/budget_ablation_"$RUN_ID"_summary.csv \\
        --output-dir /tmp/rhda_budget_ablation_figures

Required CSV columns:
  - ``budget``                              (the budget label; "Unlimited" allowed)
  - ``mean_onset``                          (numeric mean predicted onset; or None)

Optional reference values:
  - ``--reference-canonical`` and ``--reference-interval`` add dashed markers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_summary_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit(f"empty summary CSV: {path}")
    return rows


def _parse_mean_onset(row: dict) -> float | None:
    v = (row.get("mean_onset") or "").strip()
    if not v or v.lower() == "none":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _budget_sort_key(label: str):
    if label == "Unlimited":
        return (1, 0)
    try:
        return (0, int(label))
    except ValueError:
        return (2, label)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, metavar="RUN_ID",
                    help="Run identifier used for labels and default output name.")
    ap.add_argument("--input-csv", type=Path, default=None,
                    help="Path to the budget summary CSV. If omitted, --input-dir + --run is used.")
    ap.add_argument("--input-dir", type=Path, default=None,
                    help="Optional directory containing budget_ablation_<run>_summary.csv.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the figure. Use an external or /tmp path.")
    ap.add_argument("--figure-name", default=None,
                    help="Output filename stem (default: budget_vs_onset_<run>).")
    ap.add_argument("--title", default=None,
                    help="Figure title (default: '<run>: budget vs mean onset').")
    ap.add_argument("--reference-canonical", type=float, default=None,
                    help="Override the canonical reference onset for the dashed line marker.")
    ap.add_argument("--reference-interval", type=str, default=None,
                    help="Override the reference interval as 'lo,hi' (used to shade a band).")
    args = ap.parse_args(argv)

    # Resolve input CSV path
    if args.input_csv is None:
        if not args.run or args.input_dir is None:
            raise SystemExit("Pass --input-csv, or pass both --input-dir and --run.")
        args.input_csv = args.input_dir / f"budget_ablation_{args.run}_summary.csv"
    if not args.input_csv.exists():
        raise SystemExit(f"input CSV not found: {args.input_csv}")

    # Resolve reference values
    canonical = args.reference_canonical
    interval: tuple[float, float] | None = None
    if args.reference_interval:
        try:
            lo, hi = (float(x) for x in args.reference_interval.split(","))
            interval = (lo, hi)
        except Exception:
            raise SystemExit("--reference-interval must be 'lo,hi' (numeric)")
    rows = _read_summary_csv(args.input_csv)
    rows = sorted(rows, key=lambda r: _budget_sort_key(r.get("budget", "")))
    labels = [r.get("budget", "?") for r in rows]
    means = [_parse_mean_onset(r) for r in rows]

    if all(m is None for m in means):
        raise SystemExit(
            f"no usable 'mean_onset' values in {args.input_csv}; cannot plot"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.figure_name or (f"budget_vs_onset_{args.run}" if args.run else "budget_vs_onset")
    out_png = args.output_dir / f"{stem}.png"
    out_pdf = args.output_dir / f"{stem}.pdf"

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    if interval is not None:
        ax.axhspan(interval[0], interval[1], color="#a0d8a0", alpha=0.30,
                   label=f"reference interval [{interval[0]:.0f}, {interval[1]:.0f}]")
    if canonical is not None:
        ax.axhline(canonical, color="#2ca02c", ls="--", lw=1.6,
                   label=f"reference canonical = {canonical:.0f}")

    valid_x = [xi for xi, m in zip(x, means) if m is not None]
    valid_m = [m for m in means if m is not None]
    ax.plot(valid_x, valid_m, marker="o", color="#1f77b4", lw=2.5, ms=10, zorder=4,
            label="mean onset")
    for i, m in enumerate(means):
        if m is not None:
            ax.text(i + 0.10, m + 8, f"{m:.0f}", fontsize=10, color="#1f77b4", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("tool-call budget", fontsize=12)
    ax.set_ylabel("mean predicted onset step", fontsize=12)
    ax.set_title(args.title or (f"{args.run}: budget vs mean onset" if args.run else "budget vs mean onset"))
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    bracket_low = min((interval[0] if interval else 1e9), min(valid_m))
    bracket_high = max((interval[1] if interval else -1e9), max(valid_m))
    ax.set_ylim(bracket_low - 40, bracket_high + 40)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
