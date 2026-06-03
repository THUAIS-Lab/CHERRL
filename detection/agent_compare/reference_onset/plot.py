#!/usr/bin/env python3
"""Plot reference-onset per-step signals from a generic full-signals JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _plot_run(ax, run_id: str, run: dict) -> None:
    rows = run.get("per_step_signals") or []
    if not rows:
        ax.set_title(f"{run_id}: no per-step signals")
        ax.axis("off")
        return
    steps = [r["step"] for r in rows]
    gap = [r.get("G_smooth") for r in rows]
    mech = [r.get("M_smooth") for r in rows]
    canonical = run.get("canonical_CO") or run.get("reference_value")
    interval = run.get("onset_uncertainty_interval") or run.get("reference_interval")
    boundary = run.get("SEP")

    ax2 = ax.twinx()
    ax.plot(steps, gap, color="#1f77b4", lw=1.2, label="G_smooth")
    ax2.plot(steps, mech, color="#2ca02c", lw=1.4, label="M_smooth (%)")
    ax.set_ylim(0, 1.0)
    ax2.set_ylim(0, 100)
    ax.set_xlabel("training step")
    ax.set_ylabel("G_smooth", color="#1f77b4")
    ax2.set_ylabel("M_smooth (%)", color="#2ca02c")

    if interval and interval[0] is not None and interval[1] is not None:
        ax.axvspan(interval[0], interval[1], color="#2ca02c", alpha=0.15)
    if canonical is not None:
        ax.axvline(canonical, color="#2ca02c", ls="--", lw=1.4)
    if boundary is not None:
        ax.axvline(boundary, color="#d62728", ls="-.", lw=1.0)

    description = run.get("mechanism_signal_description") or run.get("mechanism_signal") or ""
    ax.set_title(f"{run_id} ({description[:55]})", fontsize=9.2)
    ax.grid(True, ls=":", alpha=0.35)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-json", type=Path, required=True,
                    help="reference_onset_full_signals.json or compatible payload.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory where figures will be written.")
    ap.add_argument("--run", default=None,
                    help="Optional single run_id to plot.")
    ap.add_argument("--figure-name", default="reference_onset_signals",
                    help="Output filename stem.")
    args = ap.parse_args(argv)

    payload = json.load(args.input_json.open(encoding="utf-8"))
    runs = payload.get("runs", {})
    if args.run:
        if args.run not in runs:
            raise SystemExit(f"run not found: {args.run}")
        runs = {args.run: runs[args.run]}
    if not runs:
        raise SystemExit("input JSON has no runs")

    n = len(runs)
    cols = min(2, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (run_id, run) in zip(axes.flat, runs.items()):
        ax.axis("on")
        _plot_run(ax, run_id, run)

    legend = [
        Line2D([], [], color="#1f77b4", lw=1.4, label="G_smooth"),
        Line2D([], [], color="#2ca02c", lw=1.6, label="M_smooth (%)"),
        Line2D([], [], color="#2ca02c", ls="--", lw=1.4, label="reference onset"),
        Patch(color="#2ca02c", alpha=0.15, label="reference interval"),
        Line2D([], [], color="#d62728", ls="-.", lw=1.0, label="transition boundary"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=5, frameon=False, fontsize=8.6)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.figure_name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
