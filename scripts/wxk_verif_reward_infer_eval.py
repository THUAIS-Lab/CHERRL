#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a HF/vLLM model, run generation on the same VerIF parquet data, and score with verIF_a83ad23."
    )
    parser.add_argument("--model", required=True, help="Model or merged checkpoint path.")
    parser.add_argument(
        "--data",
        default=os.path.expanduser("~/data/if_prompts/train.parquet"),
        help="Input parquet path. Matches `wxk_verif_reward.sh` by default.",
    )
    parser.add_argument(
        "--output",
        default="outputs/verif_infer_eval.jsonl",
        help="Output jsonl path for prompt/response/score records.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary json path. Defaults to `<output>.summary.json`.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size.")
    parser.add_argument("--limit", type=int, default=-1, help="Only process the first N prompts.")
    parser.add_argument(
        "--random-sample-size",
        type=int,
        default=-1,
        help="Randomly sample N prompts from the dataset before inference.",
    )
    parser.add_argument("--n", type=int, default=1, help="Number of samples to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-base-url", default=None, help="Override the judge API base URL.")
    parser.add_argument("--judge-model-name", default=None, help="Override the judge model name.")
    parser.add_argument("--judge-api-key", default=None, help="Override the judge API key.")
    parser.add_argument("--judge-concurrency", type=int, default=16, help="Concurrent judge requests.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code to tokenizer and vLLM.",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Dataset prompt column. Training uses `prompt`.",
    )
    parser.add_argument(
        "--reward-model-key",
        default="reward_model",
        help="Dataset reward model column. Training uses `reward_model`.",
    )
    parser.add_argument(
        "--data-source-key",
        default="data_source",
        help="Dataset data source column. Training uses `data_source`.",
    )
    parser.add_argument(
        "--extra-info-key",
        default="extra_info",
        help="Dataset extra_info column. Training uses `extra_info`.",
    )
    parser.add_argument(
        "--disable-filter-overlong-prompts",
        action="store_true",
        help="Do not drop prompts longer than `--max-prompt-length`.",
    )
    return parser.parse_args()


def extract_instruction(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages)

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            parts.append(str(message))
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            continue
        if content is not None:
            parts.append(str(content))
    return "\n".join(parts).strip()


def load_dataset_rows(args: argparse.Namespace):
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=os.path.expanduser(args.data))["train"]
    if args.random_sample_size and args.random_sample_size > 0:
        sample_size = min(args.random_sample_size, len(dataset))
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(indices)
    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    return dataset


def build_generation_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_prompt_token_length(tokenizer, messages: list[dict]) -> int:
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(token_ids)


def get_ground_truth(row: dict, reward_model_key: str) -> str:
    reward_model = row.get(reward_model_key) or {}
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth", "")
    return ""


async def score_records(args: argparse.Namespace, records: list[dict]) -> None:
    from verl.utils.reward_score import verIF_a83ad23

    semaphore = asyncio.Semaphore(args.judge_concurrency)

    async def _score_one(record: dict) -> None:
        extra_info = dict(record["extra_info"])
        extra_info["raw_prompt"] = record["prompt_messages"]
        extra_info["split"] = "train"
        if args.judge_model_name:
            extra_info["model_name"] = args.judge_model_name
        if args.judge_api_key:
            extra_info["api_key"] = args.judge_api_key

        async with semaphore:
            result = await verIF_a83ad23.compute_score(
                data_source=record["data_source"],
                solution_str=record["response"],
                ground_truth=record["ground_truth"],
                extra_info=extra_info,
                reward_router_address=args.judge_base_url,
            )
        record.update(
            {
                "score": result.get("score"),
                "acc": result.get("acc"),
                "genrm_response": result.get("genrm_response"),
                "pred": result.get("pred"),
            }
        )

    await asyncio.gather(*[_score_one(record) for record in records])


def ensure_parent(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output_path = os.path.expanduser(args.output)
    summary_path = (
        os.path.expanduser(args.summary_output)
        if args.summary_output
        else os.path.expanduser(f"{args.output}.summary.json")
    )
    ensure_parent(output_path)
    ensure_parent(summary_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    dataset = load_dataset_rows(args)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_response_length,
        seed=args.seed,
    )

    filtered_rows: list[dict] = []
    skipped_overlong = 0
    for dataset_index, row in enumerate(dataset):
        prompt_messages = row[args.prompt_key]
        prompt_len = get_prompt_token_length(tokenizer, prompt_messages)
        if not args.disable_filter_overlong_prompts and prompt_len > args.max_prompt_length:
            skipped_overlong += 1
            continue
        row = dict(row)
        row["_dataset_index"] = dataset_index
        row["_prompt_len"] = prompt_len
        filtered_rows.append(row)

    all_records: list[dict] = []
    for start in range(0, len(filtered_rows), args.batch_size):
        batch_rows = filtered_rows[start : start + args.batch_size]
        batch_prompts = [build_generation_prompt(tokenizer, row[args.prompt_key]) for row in batch_rows]
        outputs = llm.generate(batch_prompts, sampling_params)

        for row, output in zip(batch_rows, outputs, strict=True):
            prompt_messages = row[args.prompt_key]
            instruction = extract_instruction(prompt_messages)
            ground_truth = get_ground_truth(row, args.reward_model_key)
            extra_info = dict(row.get(args.extra_info_key) or {})
            data_source = row.get(args.data_source_key, "unknown")

            for sample_index, candidate in enumerate(output.outputs):
                all_records.append(
                    {
                        "dataset_index": row["_dataset_index"],
                        "sample_index": sample_index,
                        "data_source": data_source,
                        "prompt": instruction,
                        "prompt_len": row["_prompt_len"],
                        "response": candidate.text,
                        "ground_truth": ground_truth,
                        "extra_info": extra_info,
                        "prompt_messages": prompt_messages,
                    }
                )

    asyncio.run(score_records(args, all_records))

    score_values = [record["score"] for record in all_records if record.get("score") is not None]
    acc_values = [1.0 if record.get("acc") else 0.0 for record in all_records if record.get("acc") is not None]

    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            output_record = {
                "dataset_index": record["dataset_index"],
                "sample_index": record["sample_index"],
                "data_source": record["data_source"],
                "prompt": record["prompt"],
                "prompt_len": record["prompt_len"],
                "response": record["response"],
                "judge_output": record.get("genrm_response"),
                "score": record.get("score"),
                "acc": record.get("acc"),
                "genrm_response": record.get("genrm_response"),
                "pred": record.get("pred"),
            }
            f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "data": os.path.expanduser(args.data),
        "random_sample_size": args.random_sample_size if args.random_sample_size > 0 else None,
        "seed": args.seed,
        "num_input_prompts": len(dataset),
        "num_used_prompts": len(filtered_rows),
        "num_skipped_overlong_prompts": skipped_overlong,
        "num_generated_records": len(all_records),
        "mean_score": mean(score_values) if score_values else None,
        "mean_acc": mean(acc_values) if acc_values else None,
        "output": output_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
