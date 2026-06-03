#!/usr/bin/env python3
"""Augment HealthBench rollout JSONL with matched rubrics.

This utility matches rollout entries by prompt text against the raw HealthBench
JSONL data and writes a new JSONL file that includes:
  - prompt_id
  - example_tags
  - rubrics

It never mutates the source rollout file. It writes a sibling or user-provided
output file instead.

Examples:
  python scripts/annotate_healthbench_rubrics.py \
      --input /data/haozy/healthbench/rollout_log/Qwen3-4B_healthbench/1.jsonl \
      --output /tmp/healthbench_step1_with_rubrics.jsonl

  python scripts/annotate_healthbench_rubrics.py \
      --input /data/haozy/healthbench/rollout_log/Qwen3-4B_healthbench \
      --output-dir /tmp/healthbench_rollout_with_rubrics
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HealthBenchRecord:
    prompt_id: str | None
    example_tags: list[str]
    rubrics: list[dict[str, Any]]


def _prompt_messages_to_rollout_key(prompt: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in prompt:
        role = message.get("role", "")
        content = message.get("content", "")
        parts.append(f"{role}\n{content}")
    parts.append("assistant\n")
    return "\n".join(parts)


def load_healthbench_index(raw_paths: list[Path]) -> dict[str, HealthBenchRecord]:
    index: dict[str, HealthBenchRecord] = {}
    for path in raw_paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                prompt = row.get("prompt")
                if not isinstance(prompt, list):
                    continue
                key = _prompt_messages_to_rollout_key(prompt)
                index.setdefault(
                    key,
                    HealthBenchRecord(
                        prompt_id=row.get("prompt_id"),
                        example_tags=list(row.get("example_tags", [])),
                        rubrics=list(row.get("rubrics", [])),
                    ),
                )
    return index


def annotate_file(input_path: Path, output_path: Path, index: dict[str, HealthBenchRecord]) -> dict[str, int]:
    matched = 0
    missed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            key = row.get("input", "")
            record = index.get(key)
            if record is None:
                row["healthbench_prompt_id"] = None
                row["healthbench_example_tags"] = []
                row["healthbench_rubrics"] = []
                row["healthbench_rubrics_matched"] = False
                missed += 1
            else:
                row["healthbench_prompt_id"] = record.prompt_id
                row["healthbench_example_tags"] = record.example_tags
                row["healthbench_rubrics"] = record.rubrics
                row["healthbench_rubrics_matched"] = True
                matched += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"matched": matched, "missed": missed}


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = [path for path in input_path.glob("*.jsonl") if path.is_file()]
    return sorted(files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def default_output_for_file(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.with_rubrics.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write matched HealthBench rubrics into rollout JSONL")
    parser.add_argument("--input", required=True, help="Input HealthBench rollout JSONL file or directory")
    parser.add_argument("--output", help="Output JSONL path when input is a single file")
    parser.add_argument("--output-dir", help="Output directory when input is a rollout directory")
    parser.add_argument(
        "--raw-jsonl",
        nargs="*",
        default=[
            "data/health_bench/raw/healthbench_train.jsonl",
            "data/health_bench/raw/healthbench_eval.jsonl",
        ],
        help="HealthBench raw JSONL sources used for prompt->rubrics matching",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    raw_paths = [Path(path) for path in args.raw_jsonl]
    index = load_healthbench_index(raw_paths)

    files = iter_input_files(input_path)
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {input_path}")

    if input_path.is_file():
        output_path = Path(args.output) if args.output else default_output_for_file(input_path)
        stats = annotate_file(input_path, output_path, index)
        print(
            json.dumps(
                {
                    "input": str(input_path),
                    "output": str(output_path),
                    "matched": stats["matched"],
                    "missed": stats["missed"],
                    "coverage": round(stats["matched"] / (stats["matched"] + stats["missed"]), 4)
                    if (stats["matched"] + stats["missed"])
                    else 0.0,
                    "index_size": len(index),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_path.with_name(f"{input_path.name}_with_rubrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_matched = 0
    total_missed = 0
    per_file: list[dict[str, Any]] = []
    for file_path in files:
        out_path = output_dir / file_path.name
        stats = annotate_file(file_path, out_path, index)
        total_matched += stats["matched"]
        total_missed += stats["missed"]
        per_file.append(
            {
                "input": str(file_path),
                "output": str(out_path),
                "matched": stats["matched"],
                "missed": stats["missed"],
            }
        )

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "files": len(files),
                "total_matched": total_matched,
                "total_missed": total_missed,
                "coverage": round(total_matched / (total_matched + total_missed), 4)
                if (total_matched + total_missed)
                else 0.0,
                "index_size": len(index),
                "per_file_preview": per_file[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
