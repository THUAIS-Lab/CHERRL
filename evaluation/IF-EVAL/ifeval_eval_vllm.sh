#!/usr/bin/env bash
set -euo pipefail

MODEL=""
CKPT_DIR=""
MERGE_TARGET=""
RESPONSES=""
INPUT_DATA=""
EVAL_OUTPUT_DIR=""
INPUT_DATA_SET=0
EVAL_OUTPUT_SET=0
TP_SIZE=4
BATCH_SIZE=128
MAX_TOKENS=4096
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.95
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
CUDA_DEVICES=""
NO_CHAT_TEMPLATE=0
TRUST_REMOTE_CODE=0
SKIP_NLTK_DOWNLOAD=0

usage() {
  cat <<'EOF'
Usage: ifeval_eval_vllm.sh --model <merged_model_path> [options]

Options:
  --model PATH                Merged HF model path for vLLM.
  --ckpt-dir PATH             FSDP checkpoint dir to merge (optional).
  --merge-target PATH         Target dir for merge (required if --ckpt-dir).
  --input PATH                IFEval input_data.jsonl (default: auto-detect google-research checkout).
  --responses PATH            Output jsonl with prompt/response (default: auto under eval repo).
  --eval-output-dir PATH      Where eval_results_*.jsonl are written (default: under eval repo).
  --tp-size N                 vLLM tensor parallel size (default: 4).
  --batch-size N              vLLM batch size (default: 8).
  --max-tokens N              Max new tokens (default: 2048).
  --max-model-len N           Max model len (default: 8192).
  --gpu-mem-util FLOAT        vLLM gpu mem util (default: 0.95).
  --temperature FLOAT         Sampling temperature (default: 0.7).
  --top-p FLOAT               Nucleus sampling p (default: 0.8).
  --top-k N                   Top-k sampling (default: 20).
  --cuda-devices LIST         Set CUDA_VISIBLE_DEVICES (e.g., 0,1,2,3).
  --no-chat-template          Use raw prompts without chat template.
  --trust-remote-code         Pass trust_remote_code to tokenizer/LLM.
  --skip-nltk-download        Do not attempt to download NLTK punkt.
  -h, --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"; shift 2 ;;
    --ckpt-dir)
      CKPT_DIR="$2"; shift 2 ;;
    --merge-target)
      MERGE_TARGET="$2"; shift 2 ;;
    --input)
      INPUT_DATA="$2"; INPUT_DATA_SET=1; shift 2 ;;
    --responses)
      RESPONSES="$2"; shift 2 ;;
    --eval-output-dir)
      EVAL_OUTPUT_DIR="$2"; EVAL_OUTPUT_SET=1; shift 2 ;;
    --tp-size)
      TP_SIZE="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --max-tokens)
      MAX_TOKENS="$2"; shift 2 ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util)
      GPU_MEM_UTIL="$2"; shift 2 ;;
    --temperature)
      TEMPERATURE="$2"; shift 2 ;;
    --top-p)
      TOP_P="$2"; shift 2 ;;
    --top-k)
      TOP_K="$2"; shift 2 ;;
    --cuda-devices)
      CUDA_DEVICES="$2"; shift 2 ;;
    --no-chat-template)
      NO_CHAT_TEMPLATE=1; shift 1 ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=1; shift 1 ;;
    --skip-nltk-download)
      SKIP_NLTK_DOWNLOAD=1; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_REPO="$SCRIPT_DIR"
# Make `instruction_following_eval` importable (symlinked as evaluation/instruction_following_eval -> IF-EVAL)
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

if [[ "$INPUT_DATA_SET" -eq 0 ]]; then
  INPUT_DATA="${EVAL_REPO}/data/input_data.jsonl"
fi
if [[ "$EVAL_OUTPUT_SET" -eq 0 ]]; then
  EVAL_OUTPUT_DIR="${EVAL_REPO}/data"
fi

if [[ -n "$CKPT_DIR" ]]; then
  if [[ -z "$MERGE_TARGET" ]]; then
    echo "--merge-target is required when using --ckpt-dir" >&2
    exit 1
  fi
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CKPT_DIR" \
    --target_dir "$MERGE_TARGET"
  MODEL="$MERGE_TARGET"
fi

if [[ -z "$MODEL" ]]; then
  echo "--model is required (or use --ckpt-dir + --merge-target)" >&2
  exit 1
fi

if [[ ! -f "$INPUT_DATA" ]]; then
  echo "input_data.jsonl not found: $INPUT_DATA" >&2
  exit 1
fi

if [[ -z "$RESPONSES" ]]; then
  tag=$(basename "$MODEL")
  RESPONSES="${EVAL_OUTPUT_DIR}/input_response_data_${tag}.jsonl"
fi

if [[ "$SKIP_NLTK_DOWNLOAD" -eq 0 ]]; then
  python - <<'PY'
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab/english")
except LookupError:
    nltk.download("punkt_tab")
PY
fi

GEN_ARGS=(
  --model "$MODEL"
  --input "$INPUT_DATA"
  --output "$RESPONSES"
  --tp-size "$TP_SIZE"
  --batch-size "$BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-mem-util "$GPU_MEM_UTIL"
)

if [[ "$NO_CHAT_TEMPLATE" -eq 1 ]]; then
  GEN_ARGS+=(--no-chat-template)
fi
if [[ "$TRUST_REMOTE_CODE" -eq 1 ]]; then
  GEN_ARGS+=(--trust-remote-code)
fi

if [[ -n "$CUDA_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

python "${SCRIPT_DIR}/ifeval_generate_vllm.py" "${GEN_ARGS[@]}"

python "${SCRIPT_DIR}/evaluation_main.py" \
  --input_data "$INPUT_DATA" \
  --input_response_data "$RESPONSES" \
  --output_dir "$EVAL_OUTPUT_DIR"

echo "Responses: $RESPONSES"
echo "Eval outputs: $EVAL_OUTPUT_DIR/eval_results_strict.jsonl, $EVAL_OUTPUT_DIR/eval_results_loose.jsonl"
