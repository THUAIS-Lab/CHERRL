#!/usr/bin/env python3
"""Build reference-onset summary artifacts from external judge-side data.

This release does not ship raw judge-side rollouts. For full recomputation,
provide an external compute script from the data release, or provide a cached
JSON produced by that external computation. This wrapper writes public artifact
filenames without embedding internal workspace names.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


def _import_external(path: Path):
    spec = importlib.util.spec_from_file_location("external_reference_onset_compute", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import external compute script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compute_with_external(script: Path, input_root: Path) -> dict[str, Any]:
    mod = _import_external(script)
    if hasattr(mod, "compute_all"):
        return mod.compute_all(input_root)
    if hasattr(mod, "DEFAULT_RUNS") and hasattr(mod, "compute_run"):
        out = {
            "kind": "reference_onset",
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "runs": {},
        }
        for cfg in mod.DEFAULT_RUNS:
            rollout = Path(cfg.get("rollout", ""))
            if not rollout.is_absolute():
                rollout = input_root / rollout
            result = mod.compute_run(cfg["run_id"], rollout, cfg["primary"])
            out["runs"][cfg["run_id"]] = result
        return out
    raise SystemExit(
        "external compute script must expose compute_all(input_root) or "
        "DEFAULT_RUNS plus compute_run(run_id, rollout_path, primary_signal)"
    )


def _load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.cache_json:
        with args.cache_json.open(encoding="utf-8") as f:
            return json.load(f)
    if args.external_compute_script:
        if not args.input_root:
            raise SystemExit("--input-root is required with --external-compute-script")
        return _compute_with_external(args.external_compute_script, args.input_root)
    raise SystemExit("pass --cache-json or --external-compute-script")


def _write_reference_json(payload: dict[str, Any], path: Path) -> None:
    trimmed = json.loads(json.dumps(payload))
    trimmed["kind"] = "reference_onset"
    for run in trimmed.get("runs", {}).values():
        run.pop("per_step_signals", None)
    path.write_text(json.dumps(trimmed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


def _write_full_json(payload: dict[str, Any], path: Path) -> None:
    full = json.loads(json.dumps(payload))
    full["kind"] = "reference_onset_full_signals"
    path.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


def _write_threshold_grid(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "label", "mechanism_signal", "gap_thr", "mech_pct", "CO"])
        for run_id, run in payload.get("runs", {}).items():
            for cell in run.get("sweep_12", []):
                writer.writerow([
                    run_id,
                    run.get("label", run_id),
                    run.get("mechanism_signal"),
                    cell.get("gap_thr"),
                    cell.get("mech_pct"),
                    cell.get("CO"),
                ])
    print(f"wrote {path}")


def _write_per_step(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "label", "step", "G_smooth", "M_smooth", "high_n"])
        for run_id, run in payload.get("runs", {}).items():
            for row in run.get("per_step_signals", []):
                writer.writerow([
                    run_id,
                    run.get("label", run_id),
                    row.get("step"),
                    row.get("G_smooth"),
                    row.get("M_smooth"),
                    row.get("high_n"),
                ])
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", type=Path,
                    help="Root containing external judge-side rollout data.")
    ap.add_argument("--external-compute-script", type=Path,
                    help="External data-release script that computes reference-onset payloads.")
    ap.add_argument("--cache-json", type=Path,
                    help="Cached full reference-onset JSON from an external computation.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory where reference-onset artifacts will be written.")
    args = ap.parse_args(argv)

    payload = _load_payload(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_reference_json(payload, args.output_dir / "reference_onset.json")
    _write_full_json(payload, args.output_dir / "reference_onset_full_signals.json")
    _write_threshold_grid(payload, args.output_dir / "threshold_grid.csv")
    _write_per_step(payload, args.output_dir / "per_step_signals.csv")


if __name__ == "__main__":
    main()
