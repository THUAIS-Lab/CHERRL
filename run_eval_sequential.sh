#!/bin/bash

# 脚本用法: 
#   ./run_eval_sequential.sh <model_name> [task1,task2,...]
# 
# 示例:
#   ./run_eval_sequential.sh qwen25_WritingBench_epoch2
#   ./run_eval_sequential.sh qwen25_WritingBench_epoch2 alpaca-eval,ifeval
#   ./run_eval_sequential.sh qwen25_WritingBench_epoch2 all
#
# 后台运行（推荐）:
#   nohup ./run_eval_sequential.sh <model_name> [task_list] > run_eval.log 2>&1 &
#   或者:
#   ./run_eval_sequential.sh <model_name> [task_list] > run_eval.log 2>&1 &

set -e

# 忽略 SIGHUP 和 SIGINT 信号，确保后台运行时不会被中断
# 注意：如果在前台运行，Ctrl+C 仍然会中断，建议使用 nohup ... & 的方式运行
trap '' HUP INT TERM

# 检查参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供模型名作为参数"
    echo ""
    echo "用法: $0 <model_name> [task_list]"
    echo ""
    echo "参数:"
    echo "  model_name  - 模型名称（必需）"
    echo "  task_list   - 要执行的任务列表，用逗号分隔（可选）"
    echo "                可用任务: alpaca-eval, ifeval, healthbench, writingbench, arena-hard"
    echo "                默认: 执行所有任务"
    echo ""
    echo "示例:"
    echo "  $0 qwen25_WritingBench_epoch2"
    echo "  $0 qwen25_WritingBench_epoch2 alpaca-eval,ifeval"
    echo "  $0 qwen25_WritingBench_epoch2 writingbench"
    exit 1
fi

MODEL_NAME="$1"
TASK_LIST="${2:-all}"

# 基础配置
# 设置代理（huggingface_hub 使用 HTTP_PROXY 和 HTTPS_PROXY）
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
BASE_URL="http://localhost:8000/v1"
JUDGE_MODEL="qwen-plus"
JUDGE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
NUM_THREADS=20
JUDGE_API_KEY="sk-0b79d17d71c147c79e708ab38d42154f"

# 日志文件后缀（可在脚本内修改，例如：_v2、_test 等）
# 例如设为 "_v2" 时，日志名会变成 eval_alpaca_v2.log 等
LOG_SUFFIX="_v2"

# 定义所有可用任务
declare -A TASKS
TASKS["alpaca-eval"]="alpaca-eval $MODEL_NAME eval_alpaca${LOG_SUFFIX}.log"
TASKS["ifeval"]="ifeval $MODEL_NAME eval_ifeval${LOG_SUFFIX}.log"
TASKS["healthbench"]="healthbench $MODEL_NAME eval_healthbench${LOG_SUFFIX}.log"
TASKS["writingbench"]="writingbench $MODEL_NAME eval_writingbench${LOG_SUFFIX}.log"
TASKS["arena-hard"]="arena-hard $MODEL_NAME eval_arena-hard${LOG_SUFFIX}.log"

# 解析任务列表
if [ "$TASK_LIST" = "all" ]; then
    SELECTED_TASKS=("alpaca-eval" "ifeval" "healthbench" "arena-hard" "writingbench")
else
    IFS=',' read -ra SELECTED_TASKS <<< "$TASK_LIST"
fi

# 验证任务是否存在
VALID_TASKS=()
for task in "${SELECTED_TASKS[@]}"; do
    task=$(echo "$task" | xargs)  # 去除空格
    if [[ -v TASKS["$task"] ]]; then
        VALID_TASKS+=("$task")
    else
        echo "警告: 未知任务 '$task'，将跳过"
        echo "可用任务: ${!TASKS[@]}"
    fi
done

if [ ${#VALID_TASKS[@]} -eq 0 ]; then
    echo "错误: 没有有效的任务可执行"
    exit 1
fi

echo "=========================================="
echo "开始按顺序执行评估任务"
echo "模型: $MODEL_NAME"
echo "任务列表: ${VALID_TASKS[*]}"
echo "总共 ${#VALID_TASKS[@]} 个任务"
echo "=========================================="
echo ""

# 执行任务
TASK_NUM=0
for task_key in "${VALID_TASKS[@]}"; do
    TASK_NUM=$((TASK_NUM + 1))
    
    # 解析任务信息: task_name model_name log_file
    read -r task_name task_model log_file <<< "${TASKS[$task_key]}"
    
    echo "[$TASK_NUM/${#VALID_TASKS[@]}] 启动 $task_name 评估任务 (模型: $task_model)..."
    echo "  日志: $log_file"
    
    python -m evaluation.eval_framework \
      --task "$task_name" \
      --model "$task_model" \
      --base-url "$BASE_URL" \
      --judge-model "$JUDGE_MODEL" \
      --judge-base-url "$JUDGE_BASE_URL" \
      --num-threads "$NUM_THREADS" \
      --judge-api-key "$JUDGE_API_KEY" \
      > "$log_file" 2>&1
    
    echo "  任务完成"
    echo ""
done

echo "=========================================="
echo "所有评估任务已完成！"
echo "日志文件:"
for task_key in "${VALID_TASKS[@]}"; do
    read -r _ _ log_file <<< "${TASKS[$task_key]}"
    echo "  - $log_file"
done
