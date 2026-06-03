#!/usr/bin/env python3
"""
Preprocess AdvancedIF CSV to parquet format for verl GRPO training.

Input: data/AdvancedIF/if_oss_full_data.csv
Output: train.parquet, val.parquet (optional split)

Schema matches verl RLHFDataset:
  - data_source: str
  - prompt: list of {role, content} (conversation messages, model input)
  - reward_model: {style, ground_truth}
  - extra_info: {split, index, conversation_history, prompt_metadata, benchmark_name}
"""

import argparse
import json
import os

import datasets
import pandas as pd
from tqdm import tqdm


# Map CSV benchmark_name to AdvancedIF judge task names
BENCHMARK_MAP = {
    "complex_if_single_turn_v5": "if_complex_if_oss",
    "if_complex_if_oss": "if_complex_if_oss",
    "if_carried_context_oss": "if_carried_context_oss",
    "if_system_steerability_oss": "if_system_steerability_oss",
}


def parse_conversation(conv_str: str) -> list:
    """Parse conversation_history from CSV string."""
    if not conv_str or not str(conv_str).strip():
        return []
    try:
        data = json.loads(conv_str)
        if isinstance(data, list):
            return [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in data]
        if isinstance(data, dict):
            return [{"role": data.get("role", "user"), "content": str(data.get("content", ""))}]
        return []
    except json.JSONDecodeError:
        return [{"role": "user", "content": str(conv_str)}]


def parse_prompt_metadata(meta_str: str) -> dict:
    """Parse prompt_metadata from CSV string."""
    if not meta_str or not str(meta_str).strip():
        return {}
    try:
        data = json.loads(meta_str)
        if isinstance(data, dict):
            return data
        return {}
    except json.JSONDecodeError:
        return {}


def parse_rubrics(meta: dict) -> list:
    """Extract rubrics list from prompt_metadata."""
    r = meta.get("rubrics")
    if r is None:
        return []
    if isinstance(r, list):
        return [str(x) for x in r]
    if isinstance(r, str):
        try:
            parsed = json.loads(r)
            return [str(x) for x in parsed] if isinstance(parsed, list) else [str(parsed)]
        except json.JSONDecodeError:
            return [r] if r.strip() else []
    return []


def main():
    parser = argparse.ArgumentParser(description="Convert AdvancedIF CSV to parquet for verl")
    parser.add_argument(
        "--input",
        "-i",
        default="data/AdvancedIF/if_oss_full_data.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="data/AdvancedIF",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation split ratio (0 = no val split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Max samples to include (-1 = all)",
    )
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "conversation_history" not in df.columns or "prompt_metadata" not in df.columns:
        raise ValueError(f"CSV must have conversation_history and prompt_metadata columns. Got: {list(df.columns)}")

    data_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        conv_str = row.get("conversation_history", "")
        meta_str = row.get("prompt_metadata", "")
        benchmark_raw = row.get("benchmark_name", "complex_if_single_turn_v5")

        conv = parse_conversation(conv_str)
        meta = parse_prompt_metadata(meta_str)
        rubrics = parse_rubrics(meta)
        benchmark = BENCHMARK_MAP.get(str(benchmark_raw), "if_complex_if_oss")

        if not conv:
            continue
        if not rubrics:
            continue

        # prompt = conversation_history (model input, without last assistant turn)
        prompt = conv
        ground_truth = json.dumps({"rubrics": rubrics})

        data_list.append({
            "data_source": "facebook/AdvancedIF",
            "prompt": prompt,
            "ability": "if",
            "reward_model": {
                "style": "rm",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "conversation_history": conv,
                "prompt_metadata": meta,
                "benchmark_name": benchmark,
            },
        })

    if args.max_samples > 0 and len(data_list) > args.max_samples:
        import random
        random.Random(args.seed).shuffle(data_list)
        data_list = data_list[: args.max_samples]

    os.makedirs(args.output_dir, exist_ok=True)

    if args.val_ratio > 0 and len(data_list) > 1:
        import random
        random.Random(args.seed).shuffle(data_list)
        n_val = max(1, int(len(data_list) * args.val_ratio))
        val_list = data_list[:n_val]
        train_list = data_list[n_val:]
        for item in val_list:
            item["extra_info"]["split"] = "val"
        train_ds = datasets.Dataset.from_list(train_list)
        val_ds = datasets.Dataset.from_list(val_list)
        train_path = os.path.join(args.output_dir, "train.parquet")
        val_path = os.path.join(args.output_dir, "val.parquet")
        train_ds.to_parquet(train_path)
        val_ds.to_parquet(val_path)
        print(f"Saved train: {train_path} ({len(train_list)} samples)")
        print(f"Saved val:   {val_path} ({len(val_list)} samples)")
    else:
        train_ds = datasets.Dataset.from_list(data_list)
        train_path = os.path.join(args.output_dir, "train.parquet")
        train_ds.to_parquet(train_path)
        print(f"Saved train: {train_path} ({len(data_list)} samples)")

    print("\nFirst example keys:", list(data_list[0].keys()))
    print("First prompt (first msg):", data_list[0]["prompt"][0]["content"][:100] + "...")


if __name__ == "__main__":
    main()
