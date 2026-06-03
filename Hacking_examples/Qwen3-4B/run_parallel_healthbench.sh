#!/bin/bash
# Batch evaluation for all HealthBench checkpoints (Qwen3-4B)
#
# Phase 1: Convert FSDP checkpoints → HuggingFace format
# Phase 2: Serve with vLLM (up to 8 GPUs parallel) + run eval-framework
#
# Usage:
#   cd /home/haozy/hackingRubricsRL
#   bash Hacking_examples/Qwen3-4B/run_parallel_healthbench.sh
#
# 后台运行:
#   nohup bash Hacking_examples/Qwen3-4B/run_parallel_healthbench.sh > eval_batch.log 2>&1 &

set -euo pipefail

# ═══════════════════════════════════════════════════
# Conda environment
# ═══════════════════════════════════════════════════
CONDA_ENV="verl"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/data/software/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "Using Python: $(which python) ($(python --version 2>&1))"

# ═══════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════
PROJECT_ROOT="/home/haozy/hackingRubricsRL"
CKPT_ROOT="/data/haozy/healthbench/checkpoints"
OUTPUT_ROOT="/data/haozy/healthbench/eval_outputs"

TASKS="healthbench,ifeval,writingbench,arena-hard,alpaca-eval"

# Judge (DASHSCOPE)
JUDGE_MODEL="qwen-flash"
JUDGE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
source "${PROJECT_ROOT}/.env" 2>/dev/null || true
JUDGE_API_KEY="${VLLM_API_KEY:-}"

# vLLM
NUM_GPUS=8
PORT_BASE=30001
MAX_MODEL_LEN=32768
NUM_THREADS=64
VLLM_READY_TIMEOUT=600          # seconds to wait per server

# ═══════════════════════════════════════════════════
# Collect checkpoints (Set A + Set B, skip Set C)
# ═══════════════════════════════════════════════════
STEPS=(50 100 150 200 250 300 350)

MODELS=()
NAMES=()

# Set A: Qwen3-4B_healthbench
for s in "${STEPS[@]}"; do
    d="${CKPT_ROOT}/Qwen3-4B_healthbench/global_step_${s}"
    if [ -d "$d/actor" ]; then
        MODELS+=("$d")
        NAMES+=("healthbench_step${s}")
    fi
done

# Set B: root-level checkpoints
for s in "${STEPS[@]}"; do
    d="${CKPT_ROOT}/global_step_${s}"
    if [ -d "$d/actor" ]; then
        MODELS+=("$d")
        NAMES+=("root_step${s}")
    fi
done

total=${#MODELS[@]}
echo "══════════════════════════════════════════════"
echo " Found $total checkpoints to evaluate"
echo "══════════════════════════════════════════════"
for idx in "${!MODELS[@]}"; do
    echo "  [${NAMES[$idx]}] ${MODELS[$idx]}"
done

if [ "$total" -eq 0 ]; then
    echo "Error: No checkpoints found!"
    exit 1
fi

# ═══════════════════════════════════════════════════
# Phase 1: Convert FSDP → HuggingFace format
# ═══════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo " Phase 1: Converting FSDP → HuggingFace"
echo "══════════════════════════════════════════════"

cd "$PROJECT_ROOT"

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python scripts/batch_actor_to_hf.py \
    "${CKPT_ROOT}/Qwen3-4B_healthbench" --keep-all-actors --skip-existing

# For Set B (root-level), convert individually to avoid recursing into subdirs
for s in "${STEPS[@]}"; do
    d="${CKPT_ROOT}/global_step_${s}"
    if [ -d "$d/actor" ] && [ ! -d "$d/actor_hf" ]; then
        echo "Converting root checkpoint: global_step_${s}"
        PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python scripts/legacy_model_merger.py \
            merge --backend fsdp \
            --local_dir "$d/actor" \
            --target_dir "$d/actor_hf"
    else
        echo "Skipping (already converted): $d"
    fi
done

# Verify all actor_hf exist
echo ""
echo "Verifying converted models..."
for idx in "${!MODELS[@]}"; do
    hf_dir="${MODELS[$idx]}/actor_hf"
    if [ ! -d "$hf_dir" ]; then
        echo "  ERROR: Missing $hf_dir"
        exit 1
    fi
    echo "  OK: ${NAMES[$idx]}"
done

# ═══════════════════════════════════════════════════
# Phase 2: Serve + Evaluate in batches of NUM_GPUS
# ═══════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo " Phase 2: vLLM Serve + Eval"
echo " Tasks: $TASKS"
echo "══════════════════════════════════════════════"

mkdir -p "$OUTPUT_ROOT"

wait_for_server() {
    local port=$1
    local deadline=$((SECONDS + VLLM_READY_TIMEOUT))
    while [ $SECONDS -lt $deadline ]; do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 3
    done
    return 1
}

cleanup_servers() {
    echo "  Cleaning up vLLM servers..."
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "${VLLM_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}

batch_num=0
for ((i = 0; i < total; i += NUM_GPUS)); do
    batch_num=$((batch_num + 1))
    end=$((i + NUM_GPUS))
    if [ $end -gt $total ]; then end=$total; fi
    batch_size=$((end - i))

    echo ""
    echo "──────────────────────────────────────────"
    echo " Batch $batch_num: models $((i+1))–${end} of $total"
    echo "──────────────────────────────────────────"

    # ── Start vLLM servers ──
    VLLM_PIDS=()
    trap cleanup_servers EXIT

    for ((j = 0; j < batch_size; j++)); do
        gpu=$j
        port=$((PORT_BASE + j))
        model_path="${MODELS[$((i + j))]}/actor_hf"
        model_name="${NAMES[$((i + j))]}"

        echo "  [GPU $gpu :$port] $model_name"

        CUDA_VISIBLE_DEVICES=$gpu vllm serve "$model_path" \
            --served-model-name "$model_name" \
            --host 0.0.0.0 \
            --port "$port" \
            --tensor-parallel-size 1 \
            --max-model-len "$MAX_MODEL_LEN" \
            > "/tmp/vllm_eval_${port}.log" 2>&1 &
        VLLM_PIDS+=($!)
    done

    # ── Wait for servers ──
    echo "  Waiting for vLLM servers to be ready..."
    all_ready=true
    for ((j = 0; j < batch_size; j++)); do
        port=$((PORT_BASE + j))
        if wait_for_server "$port"; then
            echo "    Port $port ✓"
        else
            echo "    Port $port TIMEOUT — check /tmp/vllm_eval_${port}.log"
            all_ready=false
        fi
    done

    if ! $all_ready; then
        echo "  WARNING: Some servers failed to start, continuing with available ones"
    fi

    # ── Run all tasks (--tasks handles multi-task in one call) ──
    EVAL_PIDS=()
    EVAL_NAMES=()
    for ((j = 0; j < batch_size; j++)); do
        port=$((PORT_BASE + j))
        model_name="${NAMES[$((i + j))]}"
        output_dir="${OUTPUT_ROOT}/${model_name}"
        mkdir -p "$output_dir"

        if ! curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "  SKIP $model_name (server not ready)"
            continue
        fi

        echo "  Eval: $model_name → $output_dir"

        python -m evaluation.eval_framework \
            --tasks "$TASKS" \
            --model "$model_name" \
            --base-url "http://localhost:${port}/v1" \
            --judge-model "$JUDGE_MODEL" \
            --judge-base-url "$JUDGE_BASE_URL" \
            --judge-api-key "$JUDGE_API_KEY" \
            --output-dir "$output_dir" \
            --num-threads "$NUM_THREADS" \
            > "${output_dir}/eval_all.log" 2>&1 &
        EVAL_PIDS+=($!)
        EVAL_NAMES+=("$model_name")
    done

    echo "  Waiting for ${#EVAL_PIDS[@]} evaluations (all tasks)..."
    fail_count=0
    for idx in "${!EVAL_PIDS[@]}"; do
        if wait "${EVAL_PIDS[$idx]}" 2>/dev/null; then
            echo "    ✓ ${EVAL_NAMES[$idx]}"
        else
            echo "    ✗ ${EVAL_NAMES[$idx]} — check ${OUTPUT_ROOT}/${EVAL_NAMES[$idx]}/eval_all.log"
            fail_count=$((fail_count + 1))
        fi
    done

    # ── Stop servers ──
    cleanup_servers
    trap - EXIT

    echo "  Batch $batch_num complete (all tasks)"
done

# ═══════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo " All evaluations complete!"
echo " Results: $OUTPUT_ROOT"
echo "══════════════════════════════════════════════"
echo ""
IFS=',' read -ra TASK_LIST <<< "$TASKS"
for task in "${TASK_LIST[@]}"; do
    echo "── $task ──"
    for name in "${NAMES[@]}"; do
        dir="${OUTPUT_ROOT}/${name}/${task}"
        if [ -f "$dir/summary.json" ] 2>/dev/null; then
            score=$(python3 -c "
import json, sys
d=json.load(open('$dir/summary.json'))
for k in ('score','overall_score','strict_acc','loose_acc','win_rate'):
    if k in d:
        print(f'{k}={d[k]}')
        sys.exit()
print(json.dumps(d, indent=None)[:100])
" 2>/dev/null || echo "parse-err")
            echo "  $name: $score"
        elif [ -d "$dir" ]; then
            echo "  $name: (check eval.log)"
        else
            echo "  $name: MISSING"
        fi
    done
    echo ""
done
