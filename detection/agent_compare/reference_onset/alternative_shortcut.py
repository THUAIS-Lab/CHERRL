#!/usr/bin/env python3
"""Measure a generic shortcut detector over a judge-blind mirror.

This helper is intentionally dataset-agnostic. It scans a sanitized mirror and
reports the prevalence of a user-provided regex over high-scoring responses per
step. It does not modify reference-onset result files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _iter_step_files(mirror_dir: Path):
    files = sorted(mirror_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no step JSONL files in {mirror_dir}")
    return files


def _score_threshold(rows: list[dict], quantile: float) -> float:
    scores = sorted(float(r.get("score") or 0.0) for r in rows)
    if not scores:
        return 0.0
    idx = max(0, min(len(scores) - 1, int(round((len(scores) - 1) * quantile))))
    return scores[idx]


def _analyze_step(path: Path, pattern: re.Pattern, high_quantile: float) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    thr = _score_threshold(rows, high_quantile)
    high = [r for r in rows if float(r.get("score") or 0.0) >= thr]
    hits = [r for r in high if pattern.search(r.get("output") or "")]
    return {
        "step": int(path.stem),
        "n_rows": len(rows),
        "high_score_threshold": thr,
        "high_n": len(high),
        "hit_n": len(hits),
        "hit_rate": round(len(hits) / len(high), 6) if high else None,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror-dir", type=Path, required=True,
                    help="Sanitized RHDA mirror directory for one run.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory where CSV/JSON outputs will be written.")
    ap.add_argument("--pattern", required=True,
                    help="Regex pattern to count in high-scoring responses.")
    ap.add_argument("--high-quantile", type=float, default=0.90,
                    help="Score quantile used to define the high-scoring bucket.")
    args = ap.parse_args(argv)

    if not 0.0 <= args.high_quantile <= 1.0:
        raise SystemExit("--high-quantile must be in [0, 1]")
    pattern = re.compile(args.pattern, re.IGNORECASE | re.DOTALL)
    rows = [_analyze_step(path, pattern, args.high_quantile) for path in _iter_step_files(args.mirror_dir)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "alternative_shortcut_signal.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "alternative_shortcut_signal.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
