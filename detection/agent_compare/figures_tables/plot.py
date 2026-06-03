#!/usr/bin/env python3
"""Generic paper-figure plotting utility for reference-onset / budget-style figures.

Reads a JSON or CSV input file produced by an upstream pipeline (such as
``compute_reference_onset.py`` or ``parse_budget_ablation.py``) and writes a
single figure to ``--output-dir``. The script is intentionally schema-light:
it expects the caller to specify the input file, the figure type, and any
optional run-specific reference values.

Usage
-----
    export RUN_ID=run_e   # any run id with the matching per-step signals file

    python -m detection.agent_compare.figures_tables.plot \\
        --input-json /path/to/per_step_signals.json \\
        --figure-type per-step-signals \\
        --run "$RUN_ID" \\
        --output-dir /tmp/rhda_figures_tables

Supported figure types
----------------------

- ``per-step-signals`` — line plot of high-bucket prevalence M_smooth vs step,
  optionally annotated with the canonical reference onset and interval.

The script does **not** assume any specific external workspace path; if
``--input-json`` (or ``--input-csv``) is omitted, it errors out cleanly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Optional per-run reference markers; consulted when --reference-* not given.
RUN_REFERENCE_DEFAULTS: dict[str, dict] = {
    "run_a": {"canonical": 478, "interval": (478, 492)},
    "run_b": {"canonical": 116, "interval": (115, 161)},
    "run_c": {"canonical": 91,  "interval": (91, 95)},
    "run_d": {"canonical": 68,  "interval": (68, 79)},
    "run_e": {"canonical": 301, "interval": (301, 443)},
    "run_f": {"canonical": 460, "interval": (460, 466)},
}


def _resolve_reference(args: argparse.Namespace) -> tuple[float | None, tuple[float, float] | None]:
    canonical = args.reference_canonical
    interval: tuple[float, float] | None = None
    if args.reference_interval:
        try:
            lo, hi = (float(x) for x in args.reference_interval.split(","))
            interval = (lo, hi)
        except Exception:
            raise SystemExit("--reference-interval must be 'lo,hi'")
    if (canonical is None or interval is None) and args.run in RUN_REFERENCE_DEFAULTS:
        d = RUN_REFERENCE_DEFAULTS[args.run]
        if canonical is None:
            canonical = float(d["canonical"])
        if interval is None:
            interval = tuple(float(x) for x in d["interval"])
    return canonical, interval


def _load_per_step_signals(path: Path) -> list[dict]:
    """Load a per-step signal series from JSON or CSV.

    Expected fields per row: step (int), one or more numeric columns
    (we plot the first numeric column found, or use --series-key)."""
    if path.suffix == ".json":
        blob = json.load(path.open())
        # Either a top-level list, or a dict with a list-valued key
        if isinstance(blob, list):
            return blob
        for v in blob.values():
            if isinstance(v, list):
                return v
        raise SystemExit(f"could not find a list of per-step records in {path}")
    rows: list[dict] = []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _figure_per_step_signals(args: argparse.Namespace) -> None:
    if not args.input_json and not args.input_csv:
        raise SystemExit("per-step-signals requires --input-json or --input-csv")
    src = args.input_json or args.input_csv
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    rows = _load_per_step_signals(src)
    if not rows:
        raise SystemExit(f"no rows in {src}")

    # Pick series key
    series_key = args.series_key
    if series_key is None:
        for k in ("M_smooth", "G_smooth", "high_n", "value", "y"):
            if any(k in r for r in rows):
                series_key = k
                break
    if series_key is None or not any(series_key in r for r in rows):
        raise SystemExit(
            "no plottable series found; specify --series-key explicitly. "
            f"Available keys in first row: {list(rows[0].keys())}"
        )

    steps: list[int] = []
    values: list[float] = []
    for r in rows:
        try:
            s = int(r.get("step"))
            v = float(r.get(series_key))
        except (TypeError, ValueError):
            continue
        steps.append(s)
        values.append(v)
    if not steps:
        raise SystemExit("no parseable rows with (step, series) found")

    canonical, interval = _resolve_reference(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.figure_name or f"per_step_signals_{args.run or 'figure'}"
    out_png = args.output_dir / f"{stem}.png"
    out_pdf = args.output_dir / f"{stem}.pdf"

    fig, ax = plt.subplots(figsize=(12, 4.5))
    if interval is not None:
        ax.axvspan(interval[0], interval[1], color="#a0d8a0", alpha=0.30,
                   label=f"reference interval [{interval[0]:.0f}, {interval[1]:.0f}]")
    if canonical is not None:
        ax.axvline(canonical, color="#2ca02c", ls="--", lw=1.4,
                   label=f"reference canonical = {canonical:.0f}")
    ax.plot(steps, values, color="#1f77b4", lw=1.6, label=series_key)
    ax.set_xlabel("training step", fontsize=12)
    ax.set_ylabel(series_key, fontsize=12)
    ax.set_title(args.title or (f"{args.run}: per-step signal" if args.run else "per-step signal"))
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


FIGURE_TYPES = {
    "per-step-signals": _figure_per_step_signals,
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure-type", default="per-step-signals", choices=tuple(FIGURE_TYPES.keys()),
                    help="Which kind of figure to render.")
    ap.add_argument("--input-json", type=Path, default=None,
                    help="Input JSON (e.g. reference_onset_full_signals.json).")
    ap.add_argument("--input-csv", type=Path, default=None,
                    help="Input CSV (e.g. per_step_signals.csv).")
    ap.add_argument("--series-key", default=None,
                    help="Column / field name to plot on y axis (default: auto-detect).")
    ap.add_argument("--run", default=None,
                    help="Run id (run_a..f); used to pull paper-canonical reference markers.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the figure. Use an external or /tmp path.")
    ap.add_argument("--figure-name", default=None,
                    help="Output filename stem (default: per_step_signals_<run>).")
    ap.add_argument("--title", default=None,
                    help="Figure title.")
    ap.add_argument("--reference-canonical", type=float, default=None,
                    help="Override canonical onset value for the marker line.")
    ap.add_argument("--reference-interval", type=str, default=None,
                    help="Override interval as 'lo,hi' for the shaded band.")
    args = ap.parse_args(argv)

    FIGURE_TYPES[args.figure_type](args)


if __name__ == "__main__":
    main()
