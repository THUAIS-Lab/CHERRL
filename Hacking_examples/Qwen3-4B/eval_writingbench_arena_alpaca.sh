#!/usr/bin/env bash
# =============================================================================
# eval_writingbench_arena_alpaca.sh
#
# Evaluates Qwen3-4B_healthbench_tone_bias checkpoints on WritingBench,
# Arena-Hard, and AlpacaEval. Results are written to the same OUT_DIR used by
# eval_healthbench.sh so that all four benchmarks can be plotted together.
#
# Full pipeline:  Inference (vLLM) → Judge (API) → Plot (all 4 tasks)
#
# Usage:
#   bash eval_writingbench_arena_alpaca.sh
#
#   Override any CONFIG field at invocation time, e.g.:
#   PHASE=inference STEPS=280 bash ...
#   PHASE=judge     STEPS="70,140,210,280" bash ...
#   SKIP_COMPLETE=1 bash ...   # resume where you left off
#   DRY_RUN=1       bash ...   # preview config and exit
# =============================================================================
set -euo pipefail

# ── SSH-disconnect guard ──────────────────────────────────────────────────────
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && command -v tmux >/dev/null 2>&1; then
  _sess="$(basename "$0" .sh)"
  echo "Not inside tmux — relaunching as session '${_sess}'."
  echo "  Reconnect later with:  tmux attach -t ${_sess}"
  exec tmux new-session -s "${_sess}" \
    "bash $(realpath -- "$0") $(printf '%q ' "$@"); echo; echo '=== Script finished. Press Enter to close. ==='; read _"
fi
# ─────────────────────────────────────────────────────────────────────────────

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

EVAL_FRAMEWORK_ROOT="/data/nvme1/wangxk/hackingRubricsRL/evaluation/eval_framework"
VENV="/data/nvme1/conda/envs/rubrics_wxk"
REPO="/data/nvme1/wangxk/hackingRubricsRL"

if [ -f "${EVAL_FRAMEWORK_ROOT}/.env" ]; then
  source "${EVAL_FRAMEWORK_ROOT}/.env"
fi

# Checkpoint directory containing global_step_*/actor_hf subdirectories
: "${CKPT_DIR:=/data/nvme1/wangxk/ckpts/Qwen3-4B_healthbench_tone_bias}"
: "${EXP_LABEL:=Qwen3-4B_healthbench_tone_bias}"

# Write results into the SAME output directory so all tasks share step_N/
: "${OUT_DIR:=/data/nvme1/wangxk/eval_outputs/${EXP_LABEL}}"
: "${PLOT_DIR:=${OUT_DIR}/plots}"

# Steps to evaluate (must match those already run for healthbench)
: "${STEPS:=70 140 210 280}"

# Tasks for this script (healthbench is already done)
: "${INFERENCE_TASKS:=writingbench,arena-hard,alpaca-eval}"
: "${JUDGE_TASKS:=writingbench,arena-hard,alpaca-eval}"

# These tasks don't benefit from multiple independent runs
: "${NUM_RUNS:=1}"

# GPU & vLLM settings
: "${GPU_IDS:=0,1}"
: "${TP_SIZE:=1}"
: "${GPU_MEM_UTIL:=0.85}"
: "${BASE_PORT:=30001}"
: "${INFERENCE_THREADS:=512}"

# Judge settings
: "${JUDGE_MODEL:=${AGENT_MODEL:-qwen-plus}}"
: "${JUDGE_BASE_URL:=${AGENT_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
: "${JUDGE_API_KEY:=${AGENT_API_KEY:-}}"
: "${JUDGE_THREADS:=32}"
: "${JUDGE_BATCH_SIZE:=5}"

: "${LOG_DIR:=/data/nvme1/wangxk/eval_logs/${EXP_LABEL}_extra_bench}"

: "${RUN_INFERENCE:=1}"
: "${RUN_JUDGE:=1}"
: "${RUN_PLOT:=1}"

: "${PHASE:=all}"
: "${SKIP_COMPLETE:=1}"
: "${DRY_RUN:=0}"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  END CONFIG                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

_to_array() {
  local name=$1 raw
  raw="${!name}"
  raw="${raw//,/ }"
  read -ra _tmp <<< "$raw"
  eval "${name}=(\"\${_tmp[@]}\")"
}

_is_array() {
  [[ "$(declare -p "$1" 2>/dev/null)" == "declare -a"* ]]
}
_is_array STEPS   || _to_array STEPS
_is_array GPU_IDS || _to_array GPU_IDS

case "${PHASE}" in
  all)                              ;;
  inference|infer|i)                RUN_INFERENCE=1; RUN_JUDGE=0; RUN_PLOT=0 ;;
  judge|j)                          RUN_INFERENCE=0; RUN_JUDGE=1; RUN_PLOT=0 ;;
  plot|p)                           RUN_INFERENCE=0; RUN_JUDGE=0; RUN_PLOT=1 ;;
  ij|inference+judge|infer+judge)   RUN_INFERENCE=1; RUN_JUDGE=1; RUN_PLOT=0 ;;
  jp|judge+plot)                    RUN_INFERENCE=0; RUN_JUDGE=1; RUN_PLOT=1 ;;
  *)
    echo "ERROR: invalid PHASE='${PHASE}'." >&2
    exit 1
    ;;
esac

export PATH="${VENV}/bin:${PATH}"
export VLLM_USE_DEEP_GEMM=0
# Point Python's urllib (used by samplers.py) to certifi's CA bundle so the
# dashscope/judge API calls don't fail with SSL_CERT_VERIFY_FAILED on hosts
# where the system CA store is unavailable.
if [ -z "${SSL_CERT_FILE:-}" ]; then
  _certifi_bundle="$("${VENV}/bin/python" -m certifi 2>/dev/null || true)"
  [ -n "${_certifi_bundle}" ] && [ -f "${_certifi_bundle}" ] && export SSL_CERT_FILE="${_certifi_bundle}"
fi
export EVAL_THROTTLE_STATE_PATH=/tmp/eval_framework_throttle_$(whoami).state
cd "${REPO}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${PLOT_DIR}"

# Auto-detect steps from checkpoint directory when STEPS is empty
if [ ${#STEPS[@]} -eq 0 ]; then
  echo "Auto-detecting steps from ${CKPT_DIR}..."
  for d in "${CKPT_DIR}"/global_step_*/; do
    [ -d "$d" ] || continue
    step=$(basename "$d" | sed 's/global_step_//')
    STEPS+=("$step")
  done
  IFS=$'\n' STEPS=($(sort -n <<<"${STEPS[*]}")); unset IFS
  echo "  Found ${#STEPS[@]} steps: ${STEPS[*]}"
fi

if [ ${#STEPS[@]} -eq 0 ]; then
  echo "ERROR: No steps found in ${CKPT_DIR}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Completion detectors
# ---------------------------------------------------------------------------
_inference_complete() {
  local step=$1 dir="${OUT_DIR}/step_${step}" t
  local IFS=','
  for t in ${INFERENCE_TASKS}; do
    case "${t}" in
      writingbench)
        [ -e "${dir}/${t}/responses/responses.jsonl" ] || return 1 ;;
      arena-hard|alpaca-eval)
        [ -e "${dir}/${t}/model_answer" ] || return 1 ;;
      *)
        [ -e "${dir}/${t}/responses.jsonl" ] || [ -e "${dir}/responses.jsonl" ] || return 1 ;;
    esac
  done
  return 0
}

_judge_complete() {
  local step=$1 dir="${OUT_DIR}/step_${step}" t
  local IFS=','
  for t in ${JUDGE_TASKS}; do
    [ -f "${dir}/${t}/summary.json" ] || return 1
  done
  return 0
}

_filter_steps() {
  local pred=$1 label=$2 out_name=$3; shift 3
  local step keep=()
  for step in "$@"; do
    if "$pred" "$step"; then
      echo "  [skip ${label}] step_${step} (artifacts already present)"
    else
      keep+=("$step")
    fi
  done
  eval "${out_name}=(\"\${keep[@]}\")"
}

declare -a PHASE1_STEPS=("${STEPS[@]}")
declare -a PHASE2_STEPS=("${STEPS[@]}")
if [ "${SKIP_COMPLETE}" = "1" ]; then
  [ "${RUN_INFERENCE}" -eq 1 ] && _filter_steps _inference_complete "infer" PHASE1_STEPS "${STEPS[@]}"
  [ "${RUN_JUDGE}"     -eq 1 ] && _filter_steps _judge_complete     "judge" PHASE2_STEPS "${STEPS[@]}"
fi

_mask() { [ -n "$1" ] && echo "<set, ${#1} chars>" || echo "<empty>"; }
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Effective config                                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
printf "  %-20s %s\n" "EXP_LABEL"        "${EXP_LABEL}"
printf "  %-20s %s\n" "CKPT_DIR"         "${CKPT_DIR}"
printf "  %-20s %s\n" "OUT_DIR"          "${OUT_DIR}"
printf "  %-20s %s\n" "PLOT_DIR"         "${PLOT_DIR}"
printf "  %-20s %s\n" "LOG_DIR"          "${LOG_DIR}"
printf "  %-20s %s (%d step(s))\n" "STEPS"  "${STEPS[*]}" "${#STEPS[@]}"
if [ "${SKIP_COMPLETE}" = "1" ]; then
  [ "${RUN_INFERENCE}" -eq 1 ] && \
    printf "  %-20s %s (%d step(s))\n" "  → phase1 inference" "${PHASE1_STEPS[*]:-<none>}" "${#PHASE1_STEPS[@]}"
  [ "${RUN_JUDGE}" -eq 1 ] && \
    printf "  %-20s %s (%d step(s))\n" "  → phase2 judge"     "${PHASE2_STEPS[*]:-<none>}" "${#PHASE2_STEPS[@]}"
fi
printf "  %-20s %s (%d gpu(s), TP=%s)\n" "GPU_IDS" "${GPU_IDS[*]}" "${#GPU_IDS[@]}" "${TP_SIZE}"
printf "  %-20s %s\n" "NUM_RUNS"         "${NUM_RUNS}"
printf "  %-20s %s\n" "INFERENCE_TASKS"  "${INFERENCE_TASKS}"
printf "  %-20s %s\n" "JUDGE_TASKS"      "${JUDGE_TASKS}"
printf "  %-20s %s\n" "JUDGE_MODEL"      "${JUDGE_MODEL}"
printf "  %-20s %s\n" "JUDGE_BASE_URL"   "${JUDGE_BASE_URL}"
printf "  %-20s %s\n" "JUDGE_API_KEY"    "$(_mask "${JUDGE_API_KEY}")"
printf "  %-20s %s  (inference=%s judge=%s plot=%s)\n" "PHASE" "${PHASE}" "${RUN_INFERENCE}" "${RUN_JUDGE}" "${RUN_PLOT}"
printf "  %-20s %s\n" "SKIP_COMPLETE"    "${SKIP_COMPLETE}"
echo ""

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1 — exiting without executing any phase."
  exit 0
fi

# ---------------------------------------------------------------------------
# Helper: serve one checkpoint, run inference, kill vLLM
# ---------------------------------------------------------------------------
serve_and_eval() {
  local gpu_list=$1 port=$2 model_path=$3 name=$4 out_dir=$5 tasks=$6

  echo "[GPU ${gpu_list}] Serve ${name} → :${port} | tasks: ${tasks}"

  CUDA_VISIBLE_DEVICES=${gpu_list} vllm serve "${model_path}" \
    --served-model-name "${name}" \
    --host 0.0.0.0 --port "${port}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    > "${LOG_DIR}/vllm_${name}.log" 2>&1 &
  local pid=$!

  local waited=0
  while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
    if ! kill -0 ${pid} 2>/dev/null; then
      echo "[GPU ${gpu_list}] FAIL: vllm died — see ${LOG_DIR}/vllm_${name}.log"
      return 1
    fi
    if (( waited >= 300 )); then
      echo "[GPU ${gpu_list}] FAIL: vllm startup timeout (${waited}s)"
      kill ${pid} 2>/dev/null || true
      return 1
    fi
    sleep 3; waited=$((waited + 3))
  done
  echo "[GPU ${gpu_list}] Ready (${waited}s). Running inference..."

  eval-framework \
    --tasks "${tasks}" \
    --model "${name}" \
    --base-url "http://localhost:${port}/v1" \
    --inference-only \
    --num-runs "${NUM_RUNS}" \
    --output-dir "${out_dir}" \
    --num-threads "${INFERENCE_THREADS}" \
    2>&1 | tee "${LOG_DIR}/eval_${name}.log"

  echo "[GPU ${gpu_list}] Done: ${name}"

  kill -- -${pid} 2>/dev/null || kill ${pid} 2>/dev/null || true
  wait ${pid} 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Phase 1: Parallel inference
# ---------------------------------------------------------------------------
if [ "${RUN_INFERENCE}" -eq 1 ]; then
  if [ ${#PHASE1_STEPS[@]} -eq 0 ]; then
    echo "Phase 1: nothing to do (all ${#STEPS[@]} step(s) already have inference artifacts)."
  else

  num_gpus=${#GPU_IDS[@]}
  if (( num_gpus % TP_SIZE != 0 )); then
    echo "ERROR: ${num_gpus} GPUs not divisible by TP_SIZE=${TP_SIZE}" >&2
    exit 1
  fi
  slots=$((num_gpus / TP_SIZE))

  declare -a SLOT_GPUS=()
  for (( s=0; s<slots; s++ )); do
    base=$((s * TP_SIZE))
    list=""
    for (( k=0; k<TP_SIZE; k++ )); do
      [ -n "${list}" ] && list+=","
      list+="${GPU_IDS[$((base + k))]}"
    done
    SLOT_GPUS+=("${list}")
  done

  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 1: Inference (${#PHASE1_STEPS[@]} checkpoints, ${num_gpus} GPUs, TP=${TP_SIZE}, slots=${slots})"
  echo "╚══════════════════════════════════════════════════════════════════╝"

  total=${#PHASE1_STEPS[@]}
  rounds=$(( (total + slots - 1) / slots ))

  for (( round=0; round<rounds; round++ )); do
    start_idx=$((round * slots))
    end_idx=$((start_idx + slots))
    (( end_idx > total )) && end_idx=${total}
    count=$((end_idx - start_idx))

    echo ""
    echo "── Round $((round+1))/${rounds}: steps ${PHASE1_STEPS[$start_idx]}..${PHASE1_STEPS[$((end_idx-1))]} (${count} jobs) ──"

    for (( j=0; j<count; j++ )); do
      idx=$((start_idx + j))
      step=${PHASE1_STEPS[$idx]}
      gpu_list=${SLOT_GPUS[$j]}
      port=$((BASE_PORT + j))
      name="s${step}"
      model_path="${CKPT_DIR}/global_step_${step}/actor_hf"
      out_dir="${OUT_DIR}/step_${step}"

      serve_and_eval "${gpu_list}" ${port} "${model_path}" "${name}" "${out_dir}" "${INFERENCE_TASKS}" &
    done

    wait
    echo "── Round $((round+1)) done. Cleaning up... ──"
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
  done

  echo ""
  echo "Phase 1 complete. Inference outputs: ${OUT_DIR}/"
  fi
fi

# ---------------------------------------------------------------------------
# Phase 2: Judge-only scoring
# ---------------------------------------------------------------------------
if [ "${RUN_JUDGE}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 2: Judge-only scoring (judge=${JUDGE_MODEL})"
  echo "╚══════════════════════════════════════════════════════════════════╝"

  if [ -z "${JUDGE_API_KEY}" ]; then
    echo "WARNING: JUDGE_API_KEY is empty. Skipping judge phase."
  else
    if [ ${#PHASE2_STEPS[@]} -eq 0 ]; then
      echo "Phase 2: nothing to do (all ${#STEPS[@]} step(s) already judged)."
    else
      judge_one() {
        local step=$1 name="s${1}"
        local dir="${OUT_DIR}/step_${step}"
        echo "[step_${step}] Judging (${JUDGE_TASKS})..."
        eval-framework \
          --tasks "${JUDGE_TASKS}" \
          --model "${name}" \
          --judge-model "${JUDGE_MODEL}" \
          --judge-base-url "${JUDGE_BASE_URL}" \
          --judge-api-key "${JUDGE_API_KEY}" \
          --output-dir "${dir}" \
          --judge-only \
          --num-runs "${NUM_RUNS}" \
          --num-threads "${JUDGE_THREADS}" \
          > "${LOG_DIR}/judge_s${step}.log" 2> >(tee -a "${LOG_DIR}/judge_s${step}.log" >&2)
        echo "[step_${step}] Judge done."
      }

      batch_num=0
      job_count=0
      total_batches=$(( (${#PHASE2_STEPS[@]} + JUDGE_BATCH_SIZE - 1) / JUDGE_BATCH_SIZE ))

      for step in "${PHASE2_STEPS[@]}"; do
        if (( job_count % JUDGE_BATCH_SIZE == 0 )); then
          (( job_count > 0 )) && wait
          batch_num=$((batch_num + 1))
          echo ""
          echo "── Judge batch ${batch_num}/${total_batches} ──"
        fi
        judge_one "${step}" &
        job_count=$((job_count + 1))
      done
      wait

      echo ""
      echo "Phase 2 complete."
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Phase 3: Plot training curves (all 4 benchmarks together)
# ---------------------------------------------------------------------------
if [ "${RUN_PLOT}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 3: Plotting (healthbench + writingbench + arena-hard + alpaca-eval)"
  echo "╚══════════════════════════════════════════════════════════════════╝"

  steps_csv=$(IFS=,; echo "${STEPS[*]}")

  python "${EVAL_FRAMEWORK_ROOT}/tools/plot_training_curves.py" \
    --runs "${EXP_LABEL}=${OUT_DIR}" \
    --name-pattern "${EXP_LABEL}=step_{step}" \
    --steps "${steps_csv}" \
    --tasks "healthbench,writingbench,arena-hard,alpaca-eval" \
    --plot-dir "${PLOT_DIR}"

  echo ""
  echo "Plots saved to: ${PLOT_DIR}/"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  All done!                                                      "
echo "║  Outputs : ${OUT_DIR}/                               "
echo "║  Plots   : ${PLOT_DIR}/                              "
echo "║  Logs    : ${LOG_DIR}/                               "
echo "╚══════════════════════════════════════════════════════════════════╝"
