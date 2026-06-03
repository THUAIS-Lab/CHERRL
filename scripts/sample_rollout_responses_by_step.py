#!/usr/bin/env python3
"""Sample rollout responses from each training step into one JSONL file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read rollout JSONL files under a directory, sample up to N responses "
            "from each step, and merge them into a single JSONL file."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-step rollout JSONL files such as 0.jsonl, 1.jsonl, ...",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for the merged sampled records.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Maximum number of responses to sample from each step. Default: 50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}, got {type(obj)}")
            records.append(obj)
    return records


def numeric_step_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), path.name)
    except ValueError:
        return (10**18, path.name)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    step_files = sorted(input_dir.glob("*.jsonl"), key=numeric_step_key)
    if not step_files:
        raise FileNotFoundError(f"No JSONL files found under {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total_written = 0
    total_steps = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for step_file in step_files:
            records = load_jsonl(step_file)
            if not records:
                continue

            sample_count = min(args.sample_size, len(records))
            sampled_records = rng.sample(records, sample_count) if len(records) > sample_count else list(records)

            try:
                step = int(step_file.stem)
            except ValueError:
                step = None

            for sample_idx, record in enumerate(sampled_records):
                merged = dict(record)
                merged["_source_file"] = step_file.name
                merged["_source_path"] = str(step_file)
                merged["_sample_index_in_step"] = sample_idx
                if step is not None and "step" not in merged:
                    merged["step"] = step
                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")

            total_written += sample_count
            total_steps += 1

    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output": str(output_path),
                "sample_size_per_step": args.sample_size,
                "seed": args.seed,
                "steps_processed": total_steps,
                "records_written": total_written,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
