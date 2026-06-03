#!/usr/bin/env python3
"""Parse generic RHDA budget-sweep workspaces into raw and summary CSVs.

The parser does not assume paper-specific workspace names. Point it at the
root that contains budgeted runs and, optionally, the root that contains
unlimited-budget runs.

Expected default layout:

    <budget-root>/budget_<budget>/rep<rep>/.../agent_alert_step*.json
    <budget-root>/budget_<budget>/rep<rep>/.../usage_summary.json
    <unlimited-root>/rep<rep>/.../agent_alert_step*.json

Use ``--budget-pattern`` / ``--unlimited-pattern`` if your restored workspace
uses a different layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _parse_csv_list(text: str, cast=str) -> list:
    out = []
    for item in (text or "").split(","):
        item = item.strip()
        if item:
            out.append(cast(item))
    return out


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _format_pattern(pattern: str, *, budget: int | str | None, rep: int) -> str:
    values = {"budget": budget, "rep": rep}
    try:
        return pattern.format(**values)
    except KeyError as exc:
        raise SystemExit(f"unknown placeholder in pattern {pattern!r}: {exc}") from exc


def _first_json(root: Path, glob_pattern: str) -> Path | None:
    matches = sorted(root.rglob(glob_pattern))
    return matches[0] if matches else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _extract_usage(workspace: Path) -> dict[str, Any]:
    usage = _read_json(_first_json(workspace, "usage_summary.json"))
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else usage
    return {
        "prompt_tokens": totals.get("prompt_tokens"),
        "completion_tokens": totals.get("completion_tokens"),
        "total_tokens": totals.get("total_tokens"),
        "n_calls_llm": totals.get("n_calls") or totals.get("n_llm_calls"),
        "wall_seconds": usage.get("wall_seconds"),
    }


def _extract_alert(workspace: Path) -> dict[str, Any]:
    alert_path = _first_json(workspace, "agent_alert_step*.json")
    alert = _read_json(alert_path)
    onset = alert.get("onset_step")
    try:
        onset = int(onset) if onset is not None else None
    except (TypeError, ValueError):
        onset = None
    return {
        "has_alert": bool(alert_path),
        "alert_file": str(alert_path) if alert_path else "",
        "onset_step": onset,
        "confidence": alert.get("confidence"),
        "hacking_type": alert.get("hacking_type"),
        "severity": alert.get("severity"),
    }


def _parse_one(root: Path, rel_pattern: str, *, budget: int | str | None, rep: int) -> dict[str, Any]:
    workspace = root / _format_pattern(rel_pattern, budget=budget, rep=rep)
    row: dict[str, Any] = {
        "budget": "Unlimited" if budget is None else str(budget),
        "rep": rep,
        "workspace": str(workspace),
        "exists": workspace.exists(),
    }
    if workspace.exists():
        row.update(_extract_alert(workspace))
        row.update(_extract_usage(workspace))
    else:
        row.update({
            "has_alert": False,
            "alert_file": "",
            "onset_step": None,
            "confidence": None,
            "hacking_type": None,
            "severity": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "n_calls_llm": None,
            "wall_seconds": None,
        })
    return row


def _mean(values: list[int]) -> float | None:
    return round(float(statistics.mean(values)), 3) if values else None


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_budget: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_budget.setdefault(row["budget"], []).append(row)

    summary = []
    for budget, group in sorted(by_budget.items(), key=lambda kv: _budget_sort_key(kv[0])):
        onsets = [int(r["onset_step"]) for r in group if r.get("onset_step") not in (None, "")]
        summary.append({
            "budget": budget,
            "n_reps": len(group),
            "n_existing": sum(1 for r in group if r.get("exists")),
            "n_alerts": sum(1 for r in group if r.get("has_alert")),
            "mean_onset": _mean(onsets),
            "min_onset": min(onsets) if onsets else None,
            "max_onset": max(onsets) if onsets else None,
            "onsets": ";".join(str(v) for v in onsets),
        })
    return summary


def _budget_sort_key(label: str):
    if label == "Unlimited":
        return (1, 0)
    try:
        return (0, int(label))
    except ValueError:
        return (2, label)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, metavar="RUN_ID",
                    help="Run identifier used only for output filenames.")
    ap.add_argument("--budget-root", type=Path, default=None, metavar="DIR",
                    help="Root containing budgeted RHDA workspaces.")
    ap.add_argument("--unlimited-root", type=Path, default=None, metavar="DIR",
                    help="Optional root containing unlimited-budget RHDA workspaces.")
    ap.add_argument("--output-dir", type=Path, required=True, metavar="DIR",
                    help="Where to write raw and summary CSVs.")
    ap.add_argument("--budget-values", default="5,10,20,30,40,50",
                    help="Comma-separated budget values to parse.")
    ap.add_argument("--reps", default="1,2,3",
                    help="Comma-separated rep ids to parse.")
    ap.add_argument("--budget-pattern", default="budget_{budget}/rep{rep}",
                    help="Relative workspace pattern under --budget-root.")
    ap.add_argument("--unlimited-pattern", default="rep{rep}",
                    help="Relative workspace pattern under --unlimited-root.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Optional JSON config overriding roots, patterns, budgets, and reps.")
    args = ap.parse_args(argv)

    cfg = _load_config(args.config)
    budget_root = Path(cfg.get("budget_root", args.budget_root)) if cfg.get("budget_root", args.budget_root) else None
    unlimited_root = Path(cfg.get("unlimited_root", args.unlimited_root)) if cfg.get("unlimited_root", args.unlimited_root) else None
    budget_values = cfg.get("budget_values", _parse_csv_list(args.budget_values, int))
    reps = cfg.get("reps", _parse_csv_list(args.reps, int))
    budget_pattern = cfg.get("budget_pattern", args.budget_pattern)
    unlimited_pattern = cfg.get("unlimited_pattern", args.unlimited_pattern)

    if not budget_root and not unlimited_root:
        raise SystemExit("pass --budget-root, --unlimited-root, or --config with at least one root")

    rows: list[dict[str, Any]] = []
    if budget_root:
        for budget in budget_values:
            for rep in reps:
                rows.append(_parse_one(budget_root, budget_pattern, budget=budget, rep=int(rep)))
    if unlimited_root:
        for rep in reps:
            rows.append(_parse_one(unlimited_root, unlimited_pattern, budget=None, rep=int(rep)))

    summary = _summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"budget_ablation_{args.run}_raw.csv"
    summary_path = args.output_dir / f"budget_ablation_{args.run}_summary.csv"
    _write_csv(raw_path, rows)
    _write_csv(summary_path, summary)
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
