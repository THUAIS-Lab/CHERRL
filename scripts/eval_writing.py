"""
评估写作模型在写作任务上的通过率（全异步 vllm API 版）。

前置：需先用 vllm 启动写作模型服务，例如：
    vllm serve /data/nvme0/model/Qwen3-4B --port 8001 --tensor-parallel-size 1

用法：
    python eval_writing.py <写作要求> [选项]

示例：
    python eval_writing.py \
        'Responses concluding with "This response addresses the question".' \
        --writer_base_url http://localhost:8001/v1 \
        --writer_model Qwen3-4B
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 提示词模板
# ──────────────────────────────────────────────────────────────────────────────

WRITER_TEMPLATE = """\
Please write a response that fully satisfies the following writing requirement.

Writing requirement:
{writing_requirement}

Begin your response directly without restating the requirement:\
"""

JUDGE_TEMPLATE = """\
请根据给定的[评分偏好]判断回复是否满足这些偏好。

[回复]
{response}

[评分偏好]
{bias_prompt}

请只判断该回复是否满足[评分偏好]中的任意一条。
如果满足任意一条，请在回答最开始输出[[1]]；如果一条也不满足，请输出[[0]]。
不用输出额外内容。\
"""

# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估写作模型通过率（全异步 vllm 版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("writing_requirement", help="写作要求（提示词）")
    parser.add_argument("--num_runs", type=int, default=100, help="评估轮数")
    parser.add_argument("--concurrency", type=int, default=32, help="最大并发请求数")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="写作模型最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="写作模型采样温度")
    parser.add_argument(
        "--writer_base_url",
        default=os.getenv("WRITER_BASE_URL", "http://localhost:8001/v1"),
        help="写作模型 vllm API 地址（或设 WRITER_BASE_URL 环境变量）",
    )
    parser.add_argument(
        "--writer_model",
        default=os.getenv("WRITER_MODEL", "Qwen3-4B"),
        help="写作模型名（或设 WRITER_MODEL 环境变量）",
    )
    parser.add_argument(
        "--base_url",
        default=os.getenv("BASE_URL", "http://localhost:8000/v1"),
        help="Judge 模型 vllm API 地址（或设 BASE_URL 环境变量）",
    )
    parser.add_argument(
        "--judge_model",
        default=os.getenv("VLLM_MODEL", "Qwen3.5-27B"),
        help="Judge 模型名（或设 VLLM_MODEL 环境变量）",
    )
    parser.add_argument("--verbose", action="store_true", help="打印每次的回答和判断")
    parser.add_argument(
        "--output_dir",
        default="output",
        help="输出根目录，每次运行在其下创建时间戳子目录（默认 output/）",
    )
    return parser.parse_args()


async def generate_response(
    client: AsyncOpenAI,
    model: str,
    writing_requirement: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": WRITER_TEMPLATE.format(writing_requirement=writing_requirement)}],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return completion.choices[0].message.content


async def judge_response(
    client: AsyncOpenAI,
    model: str,
    bias_prompt: str,
    response: str,
) -> tuple[bool, str]:
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(
            bias_prompt=bias_prompt,
            response=response,
        )}],
        max_tokens=16,
        temperature=0.0,
    )
    judgment = completion.choices[0].message.content.strip()

    if "[[1]]" in judgment:
        return True, judgment
    if "[[0]]" in judgment:
        return False, judgment

    return False, judgment  # 无法解析时保守判为不通过


async def evaluate_once(
    writer_client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    idx: int,
) -> dict:
    async with semaphore:
        started_at = datetime.now(timezone.utc).isoformat()
        record: dict = {
            "idx": idx,
            "started_at": started_at,
            "writing_requirement": args.writing_requirement,
            "writer_model": args.writer_model,
            "judge_model": args.judge_model,
            "response": None,
            "judgment": None,
            "pass": False,
            "error": None,
        }
        try:
            record["response"] = await generate_response(
                writer_client, args.writer_model,
                args.writing_requirement, args.max_new_tokens, args.temperature,
            )
            is_pass, judgment = await judge_response(
                judge_client, args.judge_model,
                args.writing_requirement, record["response"],
            )
            record["pass"] = is_pass
            record["judgment"] = judgment
        except Exception as e:
            record["error"] = str(e)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        return record


async def async_main(args: argparse.Namespace) -> None:
    writer_client = AsyncOpenAI(base_url=args.writer_base_url, api_key="dummy")
    judge_client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "results.jsonl"
    config_path = run_dir / "config.json"

    config = {
        "run_ts": run_ts,
        "writing_requirement": args.writing_requirement,
        "writer_model": args.writer_model,
        "writer_base_url": args.writer_base_url,
        "judge_model": args.judge_model,
        "judge_base_url": args.base_url,
        "num_runs": args.num_runs,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))

    print(f"写作要求    : {args.writing_requirement}")
    print(f"写作模型    : {args.writer_model}  ({args.writer_base_url})")
    print(f"Judge 模型  : {args.judge_model}  ({args.base_url})")
    print(f"轮数 / 并发 : {args.num_runs} / {args.concurrency}")
    print(f"输出目录    : {run_dir}\n")

    tasks = [
        evaluate_once(writer_client, judge_client, args, semaphore, i)
        for i in range(args.num_runs)
    ]

    passed = 0
    errors = 0
    completed = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        async for coro in tqdm(
            asyncio.as_completed(tasks),
            total=args.num_runs,
            desc="评估进度",
        ):
            record = await coro
            completed += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if record["error"]:
                errors += 1
                tqdm.write(f"  [错误] idx={record['idx']}  {record['error']}")
            else:
                if record["pass"]:
                    passed += 1
                if args.verbose:
                    status = "✓ Pass" if record["pass"] else "✗ Fail"
                    resp_preview = (record["response"] or "")[:120].replace("\n", " ")
                    judg_preview = (record["judgment"] or "")[:120].replace("\n", " ")
                    tqdm.write(f"\n  [{completed}] {status}")
                    tqdm.write(f"    回答摘要 : {resp_preview}...")
                    tqdm.write(f"    判断详情 : {judg_preview}")

    total_valid = args.num_runs - errors
    pass_rate = passed / total_valid * 100 if total_valid > 0 else 0.0

    print(f"\n{'=' * 55}")
    print(f"  评估完成")
    print(f"  总运行次数   : {args.num_runs}（有效 {total_valid}，出错 {errors}）")
    print(f"  通过次数     : {passed}")
    print(f"  不通过次数   : {total_valid - passed}")
    print(f"  通过率       : {pass_rate:.2f}%")
    summary = {
        "run_ts": run_ts,
        "total": args.num_runs,
        "valid": total_valid,
        "passed": passed,
        "failed": total_valid - passed,
        "errors": errors,
        "pass_rate": round(pass_rate, 4),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"  输出目录     : {run_dir}")
    print(f"{'=' * 55}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
