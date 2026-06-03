#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List

from vllm import LLM, SamplingParams

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_prompts(path: str) -> List[str]:
    prompts = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.lstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IFEval responses with vLLM.")
    parser.add_argument("--model", required=True, help="HF model path (merged).")
    parser.add_argument(
        "--input",
        default=os.path.join(SCRIPT_DIR, "data/input_data.jsonl"),
        help="IFEval input_data.jsonl path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output jsonl path with fields: prompt, response.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.95)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use raw prompts without applying chat template.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code to tokenizer/LLM.",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.input)

    if args.no_chat_template:
        vllm_prompts = prompts
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        vllm_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=args.trust_remote_code,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    with open(args.output, "w") as out:
        for i in range(0, len(prompts), args.batch_size):
            chunk_prompts = vllm_prompts[i : i + args.batch_size]
            outputs = llm.generate(chunk_prompts, params)
            for j, o in enumerate(outputs):
                response_text = strip_think(o.outputs[0].text)
                out.write(
                    json.dumps(
                        {"prompt": prompts[i + j], "response": response_text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
