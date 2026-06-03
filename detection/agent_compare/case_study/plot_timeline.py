#!/usr/bin/env python3
"""Plot one or more RHDA tool-call timelines from JSONL input.

Input JSONL rows must contain:

    {"case_id": "...", "timeline": [[tool_call_index, "tool_name", step_or_null], ...]}

Optional row fields: run_id, model, predicted_onset, reference_onset,
reference_interval, transition_boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mp
import matplotlib.pyplot as plt


FAMILY_STYLE = {
    "data": ("#1f77b4", "o", 28),
    "analysis": ("#2ca02c", "o", 28),
    "state": ("#888888", "s", 22),
    "terminal": ("#d62728", "*", 110),
}


def _family(tool: str) -> str:
    if tool in {"sample_cases", "read_step", "list_steps"}:
        return "data"
    if tool in {"surface_stats", "run_python", "rejudge", "top_score_correlated_tokens", "cka"}:
        return "analysis"
    if tool in {"emit_alert", "finish"}:
        return "terminal"
    return "state"


def _load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row.setdefault("case_id", row.get("label", f"case_{len(cases) + 1}"))
                cases.append(row)
    if not cases:
        raise SystemExit(f"empty timeline JSONL: {path}")
    return cases


def _as_interval(value):
    if not value:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 2:
            return None
        return (float(parts[0]), float(parts[1]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return None


def _plot_case(ax, rec: dict) -> None:
    interval = _as_interval(rec.get("reference_interval"))
    if interval:
        ax.axhspan(interval[0], interval[1], color="#2ca02c", alpha=0.18, zorder=0)
    if rec.get("reference_onset") is not None:
        ax.axhline(float(rec["reference_onset"]), color="#2ca02c", ls="--", lw=1.4, alpha=0.85)
    if rec.get("transition_boundary") is not None:
        ax.axhline(float(rec["transition_boundary"]), color="#d62728", ls="-.", lw=1.1, alpha=0.7)
    if rec.get("predicted_onset") is not None:
        ax.axhline(float(rec["predicted_onset"]), color="#ff7f0e", ls="--", lw=1.4, alpha=0.85)

    for idx, tool, step in rec.get("timeline", []):
        if step is None:
            continue
        color, marker, size = FAMILY_STYLE[_family(tool)]
        ax.scatter(idx, step, c=color, marker=marker, s=size, alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)

    title = rec["case_id"]
    details = ", ".join(str(x) for x in (rec.get("run_id"), rec.get("model")) if x)
    if details:
        title += f" ({details})"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("tool-call index", fontsize=9)
    ax.set_ylabel("step inspected", fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline-json", type=Path, required=True,
                    help="JSONL timeline data.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory where the timeline figure will be written.")
    ap.add_argument("--case-id", default=None,
                    help="Optional single case_id to plot.")
    ap.add_argument("--figure-name", default="timeline",
                    help="Output filename stem.")
    args = ap.parse_args(argv)

    cases = _load_cases(args.timeline_json)
    if args.case_id:
        cases = [c for c in cases if c.get("case_id") == args.case_id]
        if not cases:
            raise SystemExit(f"case_id not found: {args.case_id}")

    n = len(cases)
    cols = min(2, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, rec in zip(axes.flat, cases):
        ax.axis("on")
        _plot_case(ax, rec)

    scatter_handles = [
        mlines.Line2D([], [], color=FAMILY_STYLE["data"][0], marker="o", linestyle="None", markersize=7,
                      label="sample / read / list"),
        mlines.Line2D([], [], color=FAMILY_STYLE["analysis"][0], marker="o", linestyle="None", markersize=7,
                      label="analysis tools"),
        mlines.Line2D([], [], color=FAMILY_STYLE["state"][0], marker="s", linestyle="None", markersize=6,
                      label="state / workspace tools"),
        mlines.Line2D([], [], color=FAMILY_STYLE["terminal"][0], marker="*", linestyle="None", markersize=11,
                      label="emit / finish"),
        mlines.Line2D([], [], color="#2ca02c", linestyle="--", lw=1.4, label="reference onset"),
        mp.Patch(color="#2ca02c", alpha=0.18, label="reference interval"),
        mlines.Line2D([], [], color="#ff7f0e", linestyle="--", lw=1.4, label="predicted onset"),
    ]
    fig.legend(handles=scatter_handles, loc="upper center", ncol=4, frameon=True, fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.figure_name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
