#!/usr/bin/env python3
"""Plot RHDA case-study pipeline cards from CSV input.

Expected CSV columns:

    case_id,stage,title,subtitle,tools,steps,decision,status

``status`` may be ``ok`` or ``skip``. One figure is written per case.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _load_cases(path: Path) -> dict[str, list[dict]]:
    cases: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case_id = row.get("case_id") or "case"
            cases.setdefault(case_id, []).append(row)
    if not cases:
        raise SystemExit(f"empty cases CSV: {path}")
    for rows in cases.values():
        rows.sort(key=lambda r: int(r.get("stage") or 0))
    return cases


def _draw_case(case_id: str, stages: list[dict], save_to: Path) -> None:
    n = len(stages)
    fig_w = max(10, 3.2 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 96, case_id, ha="center", fontsize=14, fontweight="bold")

    box_w = min(18, 82 / max(1, n))
    gap = (100 - n * box_w) / (n + 1)
    box_top = 86
    box_bot = 8
    box_h = box_top - box_bot

    for i, stage in enumerate(stages):
        x = gap + i * (box_w + gap)
        is_skip = (stage.get("status") == "skip")
        face = "#f3f3f3" if is_skip else "#fffbe6"
        edge = "#bbbbbb" if is_skip else "#d4a017"
        color = "#aaaaaa" if is_skip else "#222222"
        ax.add_patch(FancyBboxPatch((x, box_bot), box_w, box_h,
                                    boxstyle="round,pad=0.0,rounding_size=2.2",
                                    linewidth=2, ec=edge, fc=face, zorder=2))
        y = box_top - 3
        ax.text(x + box_w / 2, y, stage.get("title", f"Stage {i + 1}"),
                ha="center", va="top", fontsize=11, fontweight="bold", color=color)
        y -= 6
        ax.text(x + box_w / 2, y, stage.get("subtitle", ""),
                ha="center", va="top", fontsize=9, style="italic", color=color)
        for label, key, dy in (("TOOLS", "tools", 7), ("STEPS", "steps", 18)):
            y -= dy
            ax.text(x + 1.2, y, label, ha="left", va="top",
                    fontsize=8.2, fontweight="bold", color=color)
            y -= 4
            ax.text(x + 1.2, y, stage.get(key, ""), ha="left", va="top",
                    fontsize=8.4, color=color)
        y -= 14
        ax.text(x + 1.2, y, "->", ha="left", va="top", fontsize=10, color=color)
        ax.text(x + 4.2, y + 0.5, stage.get("decision", ""),
                ha="left", va="top", fontsize=8.6, fontweight="bold", color=color)
        if is_skip:
            ax.text(x + box_w / 2, box_bot + box_h / 2 - 1, "SKIPPED",
                    ha="center", va="center", fontsize=18, fontweight="bold",
                    color="#cc3333", alpha=0.55, rotation=18, zorder=5)
        if i < n - 1:
            x_next = gap + (i + 1) * (box_w + gap)
            ax.add_patch(FancyArrowPatch((x + box_w, box_bot + box_h / 2),
                                         (x_next, box_bot + box_h / 2),
                                         arrowstyle="-|>", mutation_scale=18,
                                         lw=1.6, color="#666666", zorder=1))
    fig.savefig(save_to, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-csv", type=Path, required=True,
                    help="CSV containing pipeline stage rows.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory where figures will be written.")
    ap.add_argument("--case-id", default=None,
                    help="Optional single case_id to plot.")
    args = ap.parse_args(argv)

    cases = _load_cases(args.cases_csv)
    if args.case_id:
        cases = {k: v for k, v in cases.items() if k == args.case_id}
        if not cases:
            raise SystemExit(f"case_id not found: {args.case_id}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case_id, stages in cases.items():
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_id)
        out = args.output_dir / f"pipeline_{safe}.png"
        _draw_case(case_id, stages, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
