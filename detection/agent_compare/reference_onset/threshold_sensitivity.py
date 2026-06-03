#!/usr/bin/env python3
"""Run a generic threshold-sensitivity sweep over reference-onset signals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _first_meeting(rows: list[dict], gap_thr: float, mech_thr: float) -> int | None:
    for row in rows:
        gap = row.get("G_smooth")
        mech = row.get("M_smooth")
        if gap is not None and mech is not None and gap >= gap_thr and mech >= mech_thr:
            return int(row["step"])
    return None


def _mode_min(values: list[int]) -> int | None:
    if not values:
        return None
    counts = Counter(values)
    top = max(counts.values())
    return min(v for v, count in counts.items() if count == top)


def _sweep(rows: list[dict], gap_grid: tuple[float, ...], mech_grid: tuple[float, ...]) -> dict:
    cells = []
    for gap in gap_grid:
        for mech in mech_grid:
            cells.append({"gap_thr": gap, "mech_pct": mech, "CO": _first_meeting(rows, gap, mech)})
    values = [c["CO"] for c in cells if c["CO"] is not None]
    return {
        "cells": cells,
        "summary": {
            "n_cells": len(cells),
            "n_resolved": len(values),
            "canonical": _mode_min(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "interval": [min(values), max(values)] if values else [None, None],
        },
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-json", type=Path, required=True,
                    help="reference_onset_full_signals.json or compatible payload.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory where sweep JSON/CSV will be written.")
    ap.add_argument("--gap-grid", default="0.08,0.10,0.12,0.15,0.18,0.20",
                    help="Comma-separated G_smooth thresholds.")
    ap.add_argument("--mechanism-grid", default="15,20,25,30,35,40,45,50",
                    help="Comma-separated M_smooth percentage thresholds.")
    args = ap.parse_args(argv)

    payload = json.load(args.input_json.open(encoding="utf-8"))
    gap_grid = _floats(args.gap_grid)
    mech_grid = _floats(args.mechanism_grid)
    result = {"gap_grid": gap_grid, "mechanism_grid": mech_grid, "runs": {}}
    for run_id, run in payload.get("runs", {}).items():
        rows = run.get("per_step_signals", [])
        if rows:
            result["runs"][run_id] = _sweep(rows, gap_grid, mech_grid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "threshold_sensitivity.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.output_dir / "threshold_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "gap_thr", "mech_pct", "CO"])
        for run_id, run in result["runs"].items():
            for cell in run["cells"]:
                writer.writerow([run_id, cell["gap_thr"], cell["mech_pct"], cell["CO"]])
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
