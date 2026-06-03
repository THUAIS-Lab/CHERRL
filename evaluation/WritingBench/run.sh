#!/bin/bash

# 配置参数 - 请根据实际情况修改以下参数
# ============================================
# API 配置
ROLLOUT_API_KEY="sk-0b79d17d71c147c79e708ab38d42154f" # for example, you can use "sk-xxxx"
ROLLOUT_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" # for example, you can use "https://dashscope.aliyuncs.com/compatible-mode/v1"
ROLLOUT_MODEL_NAME="qwen-flash" # for example, you can use "qwen-flash" 

EVALUATOR_API_KEY="sk-0b79d17d71c147c79e708ab38d42154f" # for example, you can use "sk-xxxx"
EVALUATOR_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" # for example, you can use "https://dashscope.aliyuncs.com/compatible-mode/v1"
EVALUATOR_MODEL_NAME="qwen-flash" # for example, you can use "qwen-flash"
# 文件路径配置 
QUERY_FILE="./benchmark_query/benchmark_single_prompt.jsonl" # you can use "benchmark_single_prompt.jsonl" to test pipeline
QUERY_CRITERIA_FILE="./benchmark_query/benchmark_single_prompt.jsonl" # you can use "benchmark_single_prompt.jsonl" to test pipeline

# 注意：输出文件路径会在脚本中自动生成，使用时间戳格式
# responses/{timestamp}/response_model_{timestamp}.jsonl
# responses/{timestamp}/eval_model_{timestamp}.jsonl

# 评估器配置
EVALUATOR="claude"  # 可选: 'claude' 或 'critic'，但应该首先考虑claude格式，因为它同时也支持vllm部署的本地模型'
# ============================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 生成时间戳（格式: YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 构建文件路径
RESPONSE_OUTPUT_FILE="responses/${TIMESTAMP}/response_model_${TIMESTAMP}.jsonl"
EVAL_OUTPUT_FILE="responses/${TIMESTAMP}/eval_model_${TIMESTAMP}.jsonl"

echo "=========================================="
echo "Step 1: Running generate_response.py"
echo "=========================================="
echo "Output file: $RESPONSE_OUTPUT_FILE"

# 构建 generate_response.py 的命令
GEN_CMD="python generate_response.py"
GEN_CMD="$GEN_CMD --api_key \"$ROLLOUT_API_KEY\""
GEN_CMD="$GEN_CMD --base_url \"$ROLLOUT_BASE_URL\""
GEN_CMD="$GEN_CMD --model_name \"$ROLLOUT_MODEL_NAME\""
GEN_CMD="$GEN_CMD --output_file \"$RESPONSE_OUTPUT_FILE\""
GEN_CMD="$GEN_CMD --query_file \"$QUERY_FILE\""

# 执行 generate_response.py
echo "Executing: $GEN_CMD"
eval $GEN_CMD

# 检查执行结果
if [ $? -ne 0 ]; then
    echo "Error: generate_response.py failed!"
    exit 1
fi

# 验证输出文件是否生成
if [ ! -f "$RESPONSE_OUTPUT_FILE" ]; then
    echo "Error: Response file not found: $RESPONSE_OUTPUT_FILE"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Running evaluate_benchmark.py"
echo "=========================================="
echo "Input file: $RESPONSE_OUTPUT_FILE"
echo "Output file: $EVAL_OUTPUT_FILE"

# 构建 evaluate_benchmark.py 的命令
EVAL_CMD="python evaluate_benchmark.py"
EVAL_CMD="$EVAL_CMD --evaluator \"$EVALUATOR\""
EVAL_CMD="$EVAL_CMD --input_file \"$RESPONSE_OUTPUT_FILE\""
EVAL_CMD="$EVAL_CMD --output_file \"$EVAL_OUTPUT_FILE\""
EVAL_CMD="$EVAL_CMD --query_criteria_file \"$QUERY_CRITERIA_FILE\""
# 如果使用 claude evaluator，需要传递 API 参数
if [ "$EVALUATOR" = "claude" ]; then
    EVAL_CMD="$EVAL_CMD --api_key \"$EVALUATOR_API_KEY\""
    EVAL_CMD="$EVAL_CMD --url \"$EVALUATOR_BASE_URL\""
    EVAL_CMD="$EVAL_CMD --model_name \"$EVALUATOR_MODEL_NAME\""
fi

# 执行 evaluate_benchmark.py
echo "Executing: $EVAL_CMD"
eval $EVAL_CMD

# 检查执行结果
if [ $? -ne 0 ]; then
    echo "Error: evaluate_benchmark.py failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Response file: $RESPONSE_OUTPUT_FILE"
echo "Evaluation file: $EVAL_OUTPUT_FILE"
