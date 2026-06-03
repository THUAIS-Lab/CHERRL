#!/usr/bin/env python3
"""Project a CoT-with-score intermediate into the canonical CoT-no-score input.

The CoT monitors (``runner.py`` and ``stepwise_runner.py``) consume only
this no-score output; the with-score intermediate produced by
``build_mirror.py`` is a build-time stepping stone, not a restored
dataset.

Source row schema (intermediate, from ``build_mirror.py``):
    ``{step, row_id, input, cot, final, score}``

Output row schema (canonical CoT no-score input):
    ``{step, row_id, input, cot, final}``  (no score)

The source dataset is treated read-only. After build, the script
verifies:
  - same per-step row counts and step coverage
  - ``(step, row_id, input, cot, final)`` bit-exact match
  - ``score`` absent from every emitted row
  - no judge / reward / bias substring in any field name

Usage
-----
    export RUN_ID=my_run

    python -m detection.agent_compare.cot_monitor.build_noscore_mirror \\
        --run "$RUN_ID" \\
        --input-dir /tmp/cot_with_score_intermediate/"$RUN_ID" \\
        --output-dir detection/datasets/cot_noscore/"$RUN_ID"

The output directory is the canonical input for the CoT monitors:
``runner.py`` / ``stepwise_runner.py`` look there by default
(``detection/datasets/cot_noscore/<run_id>/<step>.jsonl`` when run from the project root).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ALLOWED_OUT_KEYS = ("step", "row_id", "input", "cot", "final")
FORBIDDEN_SUBSTRINGS = (
    "score", "genrm_response", "judge", "reward_metrics", "main_bias_pref",
    "score_diff", "gts", "main_score", "combined_score",
    "aggregated_", "aggregate_", "aux_", "acc",
)


def build_one_run(rid: str, src: Path, dst: Path, log) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    log(f"== {rid} ==")
    log(f"  src: {src}")
    log(f"  dst: {dst}")
    steps = sorted(int(p.stem) for p in src.iterdir() if p.suffix == ".jsonl")
    log(f"  steps: {len(steps)}  range {min(steps)}..{max(steps)}")
    t0 = time.time()
    total = 0
    for step in steps:
        out_rows = []
        with open(src / f"{step}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                out = {k: r.get(k) for k in ALLOWED_OUT_KEYS}
                assert set(out.keys()) == set(ALLOWED_OUT_KEYS), f"row missing/extra keys: {set(out.keys())}"
                for k in out.keys():
                    assert not any(sub in k for sub in FORBIDDEN_SUBSTRINGS), f"forbidden key in output: {k}"
                out_rows.append(out)
        with open(dst / f"{step}.jsonl", "w") as f:
            for o in out_rows:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        total += len(out_rows)
    log(f"  wrote {total} rows in {time.time()-t0:.1f}s")
    return {"run_id": rid, "src": str(src), "dst": str(dst), "n_steps": len(steps),
            "step_range": [min(steps), max(steps)], "total_rows": total}


def verify(rid: str, src: Path, dst: Path, log) -> dict:
    src_steps = sorted(int(p.stem) for p in src.iterdir() if p.suffix == ".jsonl")
    dst_steps = sorted(int(p.stem) for p in dst.iterdir() if p.suffix == ".jsonl")
    issues = []
    if src_steps != dst_steps:
        issues.append(f"step set differs: src={len(src_steps)} dst={len(dst_steps)}")
    rc_mismatch = []
    for step in src_steps:
        with open(src / f"{step}.jsonl") as f: n_src = sum(1 for _ in f)
        with open(dst / f"{step}.jsonl") as f: n_dst = sum(1 for _ in f)
        if n_src != n_dst:
            rc_mismatch.append((step, n_src, n_dst))
    if rc_mismatch:
        issues.append(f"row_count_mismatch: {rc_mismatch[:5]}")
    else:
        log(f"  per-step row count: {len(src_steps)} steps OK")
    n = len(src_steps)
    sampled = sorted({src_steps[0], src_steps[n//4], src_steps[n//2], src_steps[3*n//4], src_steps[-1]})
    matches = 0; checked = 0
    for step in sampled:
        with open(src / f"{step}.jsonl") as f: srows = [json.loads(l) for l in f]
        with open(dst / f"{step}.jsonl") as f: drows = [json.loads(l) for l in f]
        for sr, dr in zip(srows, drows):
            checked += 1
            if (sr.get("step") == dr.get("step")
                and sr.get("row_id") == dr.get("row_id")
                and sr.get("input") == dr.get("input")
                and sr.get("cot") == dr.get("cot")
                and sr.get("final") == dr.get("final")
                and "score" not in dr
                and set(dr.keys()) == set(ALLOWED_OUT_KEYS)):
                matches += 1
            else:
                if len(issues) < 10:
                    issues.append(f"step={step} row_id={dr.get('row_id')}: field-eq violation; dst keys = {sorted(dr.keys())}")
    log(f"  deep field-eq: {matches}/{checked} rows OK across sampled steps {sampled}")
    leak_steps = []
    for step in dst_steps:
        with open(dst / f"{step}.jsonl") as f:
            first = json.loads(f.readline())
        for k in first.keys():
            if any(sub in k for sub in FORBIDDEN_SUBSTRINGS):
                leak_steps.append((step, k))
                break
    if leak_steps:
        issues.append(f"forbidden-keys leaked in {len(leak_steps)} step files; first: {leak_steps[:5]}")
    else:
        log(f"  forbidden-keys leak check: 0/{len(dst_steps)} step files have a forbidden key")
    return {"run_id": rid, "sampled_steps": sampled, "deep_match_rows": matches,
            "deep_checked_rows": checked, "issues": issues,
            "score_field_absent": all(s[1] != 'score' and 'score' not in s[1] for s in leak_steps) if leak_steps else True}


def _logs_root() -> Path:
    data_root = os.environ.get("DATA_ROOT", "")
    base = (Path(data_root) / "agent_compare") if data_root else Path("/tmp/rhda_outputs_cot_monitor_logs")
    return base / "outputs_cot_monitor" / "build_cot_noscore" / "logs"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, metavar="RUN_ID",
                    help="Run identifier (any string; used as a directory label).")
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Path to the source CoT-with-score intermediate "
                         "(produced by build_mirror.py).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the canonical CoT no-score input "
                         "(typically detection/datasets/cot_noscore/<run_id>).")
    args = ap.parse_args(argv)

    log_root = _logs_root()
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"build_{args.run}.log"
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    log(f"build_noscore_mirror.py started at {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    summary = {"runs": []}
    try:
        stats = build_one_run(args.run, args.input_dir, args.output_dir, log)
        ver = verify(args.run, args.input_dir, args.output_dir, log)
        summary["runs"].append({**stats, "verify": ver})
    except Exception as e:
        log(f"  ERROR building {args.run}: {type(e).__name__}: {e}")
        summary["runs"].append({"run_id": args.run, "error": str(e)})

    (log_root / f"build_summary_{args.run}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    log("DONE")
    log_f.close()


if __name__ == "__main__":
    main()
