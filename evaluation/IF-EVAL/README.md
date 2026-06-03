# IFEval: Instruction Following Eval

This is not an officially supported Google product.

This repository contains source code and data for
[Instruction Following Evaluation for Large Language Models](arxiv.org/abs/2311.07911)


---

## vLLM-based Evaluation Scripts

We provide a vLLM-based one-click script to generate IFEval responses and compute metrics.

### 0) Install IFEval dependencies

```bash
pip install absl-py langdetect nltk immutabledict
python - <<'PY'
import nltk; nltk.download("punkt")
PY
```

### 1) Generate + Evaluate (merged model)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./ifeval_eval_vllm.sh \
  --model /data/haozy/merged_models/Qwen2.5-7B-Instruct_xxx \
  --input ./data/input_data.jsonl \
  --responses ./data/input_response_data_output.jsonl
```

### 2) (Optional) Merge FSDP ckpt

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./ifeval_eval_vllm.sh \
  --ckpt-dir  /data/haozy/global_step_350/actor \
  --merge-target /data/haozy/merged_models
```

### Outputs and Metric Mapping

IFEval writes:
- `eval_results_strict.jsonl`
- `eval_results_loose.jsonl`

Metric mapping:
- **Pr(S)** = prompt-level (strict)
- **Ins(S)** = instruction-level (strict)
- **Pr(L)** = prompt-level (loose)
- **Ins(L)** = instruction-level (loose)

### Scripts

- `ifeval_generate_vllm.py`: vLLM generation into IFEval jsonl
- `ifeval_eval_vllm.sh`: merge (optional) + generate + evaluate