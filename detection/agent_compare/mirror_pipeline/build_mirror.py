#!/usr/bin/env python3
"""Build a sanitized agent-visible mirror from a raw RL training rollout.

The mirror keeps only the four fields the agent is allowed to see:
``{step, input, output, score}``. Internal reasoning markers
(``<think>...</think>``) are removed from ``output`` so the agent only
sees the final response. ``score`` is the normalized visible aggregate
proxy score, using the run-wide ``max(|score|)`` floored at 1.0.

Usage
-----
    export RUN_ID=my_run

    python -m detection.agent_compare.mirror_pipeline.build_mirror \\
        --run "$RUN_ID" \\
        --rollout-dir /path/to/raw/"$RUN_ID"/rollout \\
        --output-dir detection/datasets/mirror/"$RUN_ID"

``--run`` accepts any string. If ``--rollout-dir`` is omitted, the script
falls back to ``$ROLLOUT_RELEASE_BASE/<paper-default>`` for paper runs
that have a registered default (currently ``run_e`` and ``run_f``); for
any other ``--run`` value, ``--rollout-dir`` is required. If
``--output-dir`` is omitted, it defaults to
``detection/datasets/mirror/<run_id>`` so RHDA can find the mirror
without further configuration.

This script reads/writes external paths only. It does **not** modify any
file in the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str | None) -> str:
    return THINK_RE.sub("", text or "")


# Paper-default raw rollout subpaths under $ROLLOUT_RELEASE_BASE.
# When --rollout-dir is not given, these are used for the listed paper runs.
# Other run ids must pass --rollout-dir explicitly.
PAPER_ROLLOUT_DEFAULTS: dict[str, str] = {
    "run_e": "merged_rollout_log/qwen3_4b_qwen_3.5-27B_verif_2gpus_with_format_bias_alpha0dot5_v2_add_agg_from_scratch",
    "run_f": "healthbench/rollout_log/Qwen3-4B_healthbench_self_praise_bias",
}

# Default output subpath under detection/datasets/mirror/<run_id>
_MODULE_FILE = Path(__file__).resolve()
_REPO_ROOT = _MODULE_FILE.parents[2]


def _default_output_dir(run: str) -> Path:
    return _REPO_ROOT / "datasets" / "mirror" / run


def _resolve_rollout_dir(run: str, arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    base = os.environ.get("ROLLOUT_RELEASE_BASE")
    sub = PAPER_ROLLOUT_DEFAULTS.get(run)
    if base and sub:
        return Path(base) / sub
    raise SystemExit(
        f"--rollout-dir not given and no paper default for run={run!r}. "
        f"Pass --rollout-dir <path> explicitly. (Paper-default fallbacks "
        f"are only registered for {sorted(PAPER_ROLLOUT_DEFAULTS)} via "
        f"$ROLLOUT_RELEASE_BASE.)"
    )


def build(rollout_dir: Path, output_dir: Path) -> None:
    if not rollout_dir.is_dir():
        raise SystemExit(f"rollout-dir does not exist: {rollout_dir}")
    files = sorted(rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no .jsonl files in {rollout_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: compute run-wide max(|score|)
    print(f"[pass 1] scanning {len(files)} files for score scale...", flush=True)
    t0 = time.time()
    max_abs = 0.0
    for f in files:
        for line in open(f):
            try:
                v = float(json.loads(line).get("score") or 0)
                if abs(v) > max_abs:
                    max_abs = abs(v)
            except Exception:
                continue
    scale = max(max_abs, 1.0)
    print(f"  scale = {scale} (in {time.time() - t0:.1f}s)", flush=True)

    # Pass 2: write sanitized 4-field rows
    print(f"[pass 2] writing mirror to {output_dir}", flush=True)
    t1 = time.time()
    total = 0
    for f in files:
        step = int(f.stem)
        out_path = output_dir / f"{step}.jsonl"
        with open(out_path, "w") as oh:
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                row = {
                    "step": step,
                    "input": (r.get("input") or "").strip(),
                    "output": _strip_think(r.get("output") or "").strip(),
                    "score": float(r.get("score") or 0.0) / scale,
                }
                oh.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    print(f"  wrote {total} rows across {len(files)} files in {time.time() - t1:.1f}s", flush=True)
    print(f"mirror at: {output_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, metavar="RUN_ID",
                    help="Run identifier. Accepts any string; used only as a directory "
                         "label. Paper-default rollout fallbacks via $ROLLOUT_RELEASE_BASE "
                         "are registered for run_e and run_f only.")
    ap.add_argument("--rollout-dir", type=Path, default=None,
                    help="Path to raw RL rollout directory (one .jsonl per step). "
                         "Required for any run id without a registered paper default.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write mirror jsonls. Default: "
                         "detection/datasets/mirror/<run_id>")
    args = ap.parse_args(argv)

    rollout_dir = _resolve_rollout_dir(args.run, str(args.rollout_dir) if args.rollout_dir else None)
    output_dir = args.output_dir or _default_output_dir(args.run)
    build(rollout_dir, output_dir)


if __name__ == "__main__":
    main()
