#!/usr/bin/env python3
"""Build a CoT-with-score mirror as an intermediate step toward CoT-no-score.

This script reads a raw rollout (one ``<step>.jsonl`` per training step)
at ``--input-dir``, extracts the ``<think>...</think>`` reasoning trace
and the ``strip_think`` final answer, re-normalizes ``score`` by run-wide
``max(|score|)``, and writes JSONL files to ``--output-dir``.

**This is a build-time intermediate, not a restored dataset.** The CoT
monitors only consume the no-score variant produced by
``build_noscore_mirror.py``, which reads the output of this script and
strips the score field. Users do not need to keep the with-score output
once the no-score mirror has been built.

Row schema (intermediate):  ``{step, row_id, input, cot, final, score}``

Strict field whitelist:
  Allowed:   step, row_id, input, cot, final, score
  FORBIDDEN: genrm_response, judge_genrm_response/*, reward_metrics/*,
             main_bias_pref, score_diff_vs_main, gts, acc, main_score,
             combined_score, aggregated_*, aggregate_*, any judge/reward/bias
             ground-truth or private mapping fields.

If ``--mirror-dir`` is provided, the script also sample-aligns the
intermediate against an existing 4-field mirror to confirm
``(input, final, score, step)`` match bit-exactly.

Usage
-----
    export RUN_ID=my_run

    python -m detection.agent_compare.cot_monitor.build_mirror \\
        --run "$RUN_ID" \\
        --input-dir /path/to/raw/"$RUN_ID"/rollout \\
        --output-dir /tmp/cot_with_score_intermediate/"$RUN_ID" \\
        [--mirror-dir detection/datasets/mirror/"$RUN_ID"]

    # Then strip score to produce the canonical CoT no-score input:
    python -m detection.agent_compare.cot_monitor.build_noscore_mirror \\
        --run "$RUN_ID" \\
        --input-dir /tmp/cot_with_score_intermediate/"$RUN_ID" \\
        --output-dir detection/datasets/cot_noscore/"$RUN_ID"

Read-only with respect to ``--input-dir`` and the existing mirror. Writes
only to ``--output-dir`` and to a logs directory under
``$DATA_ROOT/agent_compare/outputs_cot_monitor/build_cot_mirror/logs/``
(falls back to ``/tmp/rhda_outputs_cot_monitor_logs`` if ``$DATA_ROOT`` is not set).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

THINK_INNER_RE   = re.compile(r"<think>(.*?)</think>",   re.DOTALL)
THINK_REPLACE_RE = re.compile(r"<think>.*?</think>",     re.DOTALL)

ALLOWED_OUT_KEYS = ("step", "row_id", "input", "cot", "final", "score")

# These must NEVER appear in cot_mirror output rows
FORBIDDEN_KEY_SUBSTRINGS = (
    "genrm_response", "judge", "reward_metrics", "main_bias_pref",
    "score_diff", "gts", "main_score", "combined_score",
    "aggregated_", "aggregate_", "aux_", "acc",
)


def split_output(text):
    """Return (cot, final).  cot is the inside of the *first* <think>…</think>
    block (string, stripped) or None if no block.  final is strip_think(text)."""
    text = text or ""
    m = THINK_INNER_RE.search(text)
    cot = m.group(1).strip() if m else None
    final = THINK_REPLACE_RE.sub("", text).strip()
    return cot, final


def compute_score_scale(raw_dir):
    """Same convention as rhda.tool_impls._score_scale: run-wide
    max(|score|), floored at 1.0."""
    max_abs = 0.0
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".jsonl"):
            continue
        path = os.path.join(raw_dir, fn)
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    v = float(r.get("score") or 0)
                except (TypeError, ValueError):
                    continue
                if abs(v) > max_abs:
                    max_abs = abs(v)
    return max(max_abs, 1.0)


def visible_score(score, scale):
    try:
        v = float(score or 0)
    except (TypeError, ValueError):
        v = 0.0
    return round(v / scale, 6)


def build_one_run(rid, raw_dir, cot_dir, mirror_dir, log):
    cot_dir = Path(cot_dir); cot_dir.mkdir(parents=True, exist_ok=True)
    log(f"== {rid} ==")
    log(f"  raw:    {raw_dir}")
    log(f"  cot:    {cot_dir}")
    log(f"  mirror: {mirror_dir}")
    if not os.path.isdir(raw_dir):
        log(f"  ERROR: input-dir missing"); return None

    t0 = time.time()
    scale = compute_score_scale(raw_dir)
    log(f"  score_scale (run-wide max-abs): {scale}  (in {time.time()-t0:.1f}s)")

    steps = sorted(int(fn[:-6]) for fn in os.listdir(raw_dir) if fn.endswith(".jsonl"))
    log(f"  steps: {len(steps)}  range {min(steps)}..{max(steps)}")

    total_rows = 0
    cot_hit = 0
    cot_missing = 0
    cot_multi = 0       # rows whose output contains >1 <think> block
    cot_len_sum = 0
    cot_len_min = None
    cot_len_max = 0
    cot_empty_block = 0 # <think></think> with empty body

    t1 = time.time()
    for step in steps:
        raw_p = os.path.join(raw_dir, f"{step}.jsonl")
        with open(raw_p) as f:
            rows = [json.loads(l) for l in f]
        out_rows = []
        for i, r in enumerate(rows):
            inp = (r.get("input") or "").strip()
            out_text = r.get("output") or ""
            cot, final = split_output(out_text)
            if out_text.count("<think>") > 1:
                cot_multi += 1
            sc = visible_score(r.get("score", 0), scale)
            if cot is None:
                cot_missing += 1
            else:
                if cot == "":
                    cot_empty_block += 1
                else:
                    cot_hit += 1
                    L = len(cot)
                    cot_len_sum += L
                    cot_len_min = L if cot_len_min is None else min(cot_len_min, L)
                    cot_len_max = max(cot_len_max, L)
            row = {
                "step": step,
                "row_id": i,
                "input": inp,
                "cot": cot,
                "final": final,
                "score": sc,
            }
            assert set(row.keys()) <= set(ALLOWED_OUT_KEYS), f"unexpected keys: {set(row.keys())}"
            out_rows.append(row)
        total_rows += len(out_rows)
        with open(cot_dir / f"{step}.jsonl", "w") as f:
            for o in out_rows:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
    log(f"  wrote {total_rows} rows across {len(steps)} files in {time.time()-t1:.1f}s")
    return {
        "run_id": rid,
        "raw_dir": str(raw_dir),
        "cot_dir": str(cot_dir),
        "mirror_dir": str(mirror_dir) if mirror_dir else None,
        "n_steps": len(steps),
        "step_range": [min(steps), max(steps)],
        "total_rows": total_rows,
        "cot_hit": cot_hit,
        "cot_missing": cot_missing,
        "cot_empty_block": cot_empty_block,
        "cot_multi_block": cot_multi,
        "cot_hit_rate": round(cot_hit / total_rows, 6) if total_rows else 0,
        "cot_avg_chars": round(cot_len_sum / cot_hit, 1) if cot_hit else 0,
        "cot_min_chars": cot_len_min if cot_hit else 0,
        "cot_max_chars": cot_len_max,
        "score_scale": scale,
    }


def verify_alignment(rid, mirror_dir, cot_dir, log, sample_steps=None):
    """Compare cot_mirror's (step, input, final, score) against an existing
    4-field mirror's (step, input, output, score). Returns dict of stats."""
    issues = []
    matches = 0
    checked = 0
    files_full = []
    if sample_steps is None:
        steps = sorted(int(fn[:-6]) for fn in os.listdir(cot_dir) if fn.endswith(".jsonl"))
        n = len(steps)
        sample_steps = sorted({steps[0], steps[n // 4], steps[n // 2], steps[3 * n // 4], steps[-1]})
    log(f"  alignment sample_steps: {sample_steps}")

    mirror_steps = {int(fn[:-6]) for fn in os.listdir(mirror_dir) if fn.endswith(".jsonl")}
    cot_steps    = {int(fn[:-6]) for fn in os.listdir(cot_dir)    if fn.endswith(".jsonl")}
    missing_in_cot    = sorted(mirror_steps - cot_steps)
    missing_in_mirror = sorted(cot_steps - mirror_steps)
    if missing_in_cot:    issues.append(f"steps in mirror missing in cot_mirror: {missing_in_cot[:10]}{'...' if len(missing_in_cot)>10 else ''}")
    if missing_in_mirror: issues.append(f"steps in cot_mirror missing in mirror: {missing_in_mirror[:10]}{'...' if len(missing_in_mirror)>10 else ''}")
    row_count_mismatch = []
    for step in sorted(mirror_steps & cot_steps):
        with open(os.path.join(mirror_dir, f"{step}.jsonl")) as f:
            m_n = sum(1 for _ in f)
        with open(os.path.join(cot_dir, f"{step}.jsonl")) as f:
            c_n = sum(1 for _ in f)
        if m_n != c_n:
            row_count_mismatch.append((step, m_n, c_n))
    if row_count_mismatch:
        issues.append(f"row_count mismatch in {len(row_count_mismatch)} steps; first: {row_count_mismatch[:5]}")
    else:
        log(f"  per-step row count: {len(mirror_steps & cot_steps)} steps OK")

    for step in sample_steps:
        mp = os.path.join(mirror_dir, f"{step}.jsonl")
        cp = os.path.join(cot_dir,    f"{step}.jsonl")
        if not (os.path.exists(mp) and os.path.exists(cp)):
            issues.append(f"sample step {step}: missing one side"); continue
        with open(mp) as f: mrows = [json.loads(l) for l in f]
        with open(cp) as f: crows = [json.loads(l) for l in f]
        if len(mrows) != len(crows):
            issues.append(f"sample step {step}: {len(mrows)} mirror vs {len(crows)} cot rows"); continue
        for mr, cr in zip(mrows, crows):
            checked += 1
            ok_step  = mr.get("step")  == cr.get("step")
            ok_in    = mr.get("input") == cr.get("input")
            ok_out   = mr.get("output") == cr.get("final")
            ok_score = mr.get("score") == cr.get("score")
            if ok_step and ok_in and ok_out and ok_score:
                matches += 1
            else:
                if len(issues) < 10:
                    issues.append(f"mismatch step={step} row_id={cr.get('row_id')}: "
                                  f"step_eq={ok_step} input_eq={ok_in} final_eq={ok_out} score_eq={ok_score}")
        files_full.append(step)
    log(f"  deep field-eq: {matches}/{checked} rows match across sampled steps {files_full}")

    forbidden_seen = []
    with open(os.path.join(cot_dir, f"{sample_steps[0]}.jsonl")) as f:
        first = json.loads(f.readline())
    for k in first.keys():
        if any(sub in k for sub in FORBIDDEN_KEY_SUBSTRINGS):
            forbidden_seen.append(k)
    if forbidden_seen:
        issues.append(f"FORBIDDEN keys leaked: {forbidden_seen}")
    else:
        log(f"  forbidden-keys leak check: passed (only {sorted(first.keys())} present)")

    return {
        "run_id": rid,
        "sample_steps": sample_steps,
        "row_count_check_pass": not row_count_mismatch and not missing_in_cot and not missing_in_mirror,
        "deep_checked_rows": checked,
        "deep_match_rows": matches,
        "issues": issues,
    }


def _logs_root() -> Path:
    """Where build logs go. Falls back to /tmp if DATA_ROOT is unset."""
    data_root = os.environ.get("DATA_ROOT", "")
    base = (Path(data_root) / "agent_compare") if data_root else Path("/tmp/rhda_outputs_cot_monitor_logs")
    return base / "outputs_cot_monitor" / "build_cot_mirror" / "logs"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, metavar="RUN_ID",
                    help="Run identifier (any string; used as a directory label).")
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Path to the raw rollout directory (one .jsonl per step) "
                         "containing input/output/score plus <think>...</think> traces.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the cot_mirror jsonls.")
    ap.add_argument("--mirror-dir", type=Path, default=None,
                    help="Optional. If given, sample-align the new cot_mirror against an "
                         "existing 4-field mirror at this path to verify field-eq.")
    args = ap.parse_args(argv)

    log_root = _logs_root()
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"build_{args.run}.log"
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    log(f"build_cot_mirror.py started at {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    summary = {"runs": []}
    try:
        stats = build_one_run(args.run, str(args.input_dir), str(args.output_dir),
                              str(args.mirror_dir) if args.mirror_dir else None, log)
        entry = {**stats} if stats else {"run_id": args.run, "error": "build returned None"}
        if stats and args.mirror_dir:
            entry["alignment"] = verify_alignment(args.run, str(args.mirror_dir),
                                                   str(args.output_dir), log)
        summary["runs"].append(entry)
    except Exception as e:
        log(f"  ERROR building {args.run}: {type(e).__name__}: {e}")
        summary["runs"].append({"run_id": args.run, "error": str(e)})

    summary_p = log_root / f"build_summary_{args.run}.json"
    summary_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"wrote summary: {summary_p}")
    log_f.close()


if __name__ == "__main__":
    main()
