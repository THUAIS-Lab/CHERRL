#!/usr/bin/env bash
# =============================================================================
# eval_healthbench_qwen3_27B_judge_again_5runs.sh
#
# Evaluates Qwen3-4B_qwen3.5-27B_healthbench_again checkpoints on HealthBench.
# Each checkpoint is run 5 times; the framework averages the results.
#
# Full pipeline:  Inference (vLLM, --num-runs 5) → Judge (API, --num-runs 5) → Plot
#
# Usage:
#   bash eval_healthbench_qwen3_27B_judge_again_5runs.sh
#
#   Override any CONFIG field at invocation time, e.g.:
#   PHASE=inference STEPS=350 bash ...
#   PHASE=judge     STEPS="120,240,350" bash ...
#   SKIP_COMPLETE=1 bash ...   # resume where you left off
#   DRY_RUN=1 bash ...         # preview config and exit
# =============================================================================
set -euo pipefail

# ── SSH-disconnect guard ──────────────────────────────────────────────────────
# Re-launch inside a tmux session when not already in one, so the eval keeps
# running after the SSH connection drops.
#   Reconnect later with:  tmux attach -t <session-name>
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

# -- Paths --
: "${EVAL_FRAMEWORK_ROOT:=/data/nvme1/wangxk/hackingRubricsRL/evaluation/eval_framework}"
: "${VENV:=/data/nvme1/conda/envs/rubrics_wxk}"
: "${REPO:=/data/nvme1/wangxk/hackingRubricsRL}"

# Source .env early so credentials are available before overrides below.
if [ -f "${EVAL_FRAMEWORK_ROOT}/.env" ]; then
  source "${EVAL_FRAMEWORK_ROOT}/.env"
fi

# Checkpoint directory containing global_step_*/actor_hf subdirectories
: "${CKPT_DIR:=/data/wangxk/hackingRubricsRL/checkpoints/verl_grpo_healthbench/Qwen3-4B_healthbench_lexical_bias_again}"

# Experiment label (used in plot legends & log dir)
: "${EXP_LABEL:=Qwen3-4B_healthbench_lexical_bias_again}"

# Output directory: results go to ${OUT_DIR}/step_${STEP}/
: "${OUT_DIR:=/data/nvme1/wangxk/eval_outputs/${EXP_LABEL}}"

# Plot output directory
: "${PLOT_DIR:=${OUT_DIR}/plots}"

# Steps to evaluate (available: 120, 240, 350)
: "${STEPS:=70 140 210 280}"

# Tasks
: "${INFERENCE_TASKS:=healthbench}"
: "${JUDGE_TASKS:=healthbench}"

# Number of independent runs per checkpoint; results are averaged.
# Responses/scores for run i go to ${OUT_DIR}/step_${STEP}/run_i/
: "${NUM_RUNS:=3}"

# GPU & vLLM settings
: "${GPU_IDS:=0 1 2 3}"
: "${TP_SIZE:=1}"
: "${GPU_MEM_UTIL:=0.95}"
: "${BASE_PORT:=30001}"
: "${INFERENCE_THREADS:=512}"

# Judge settings (Phase 2)
: "${JUDGE_MODEL:=${AGENT_MODEL:-qwen-plus}}"
: "${JUDGE_BASE_URL:=${AGENT_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
: "${JUDGE_API_KEY:=${AGENT_API_KEY:-}}"
: "${JUDGE_THREADS:=32}"
: "${JUDGE_BATCH_SIZE:=5}"

# Logging
: "${LOG_DIR:=/data/nvme1/wangxk/eval_logs/${EXP_LABEL}}"

# Phases to run
: "${RUN_INFERENCE:=1}"
: "${RUN_JUDGE:=1}"
: "${RUN_PLOT:=1}"

# PHASE convenience knob: all | inference | judge | plot | ij | jp
: "${PHASE:=all}"

# SKIP_COMPLETE=1 drops steps that already have all phase-relevant artifacts.
: "${SKIP_COMPLETE:=1}"

# Set DRY_RUN=1 to print the effective config and exit without running.
: "${DRY_RUN:=0}"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  END CONFIG                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ---------------------------------------------------------------------------
# Normalize list-valued config into bash arrays (accept comma or space sep.)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Resolve PHASE → RUN_INFERENCE/RUN_JUDGE/RUN_PLOT
# ---------------------------------------------------------------------------
case "${PHASE}" in
  all)                              ;;
  inference|infer|i)                RUN_INFERENCE=1; RUN_JUDGE=0; RUN_PLOT=0 ;;
  judge|j)                          RUN_INFERENCE=0; RUN_JUDGE=1; RUN_PLOT=0 ;;
  plot|p)                           RUN_INFERENCE=0; RUN_JUDGE=0; RUN_PLOT=1 ;;
  ij|inference+judge|infer+judge)   RUN_INFERENCE=1; RUN_JUDGE=1; RUN_PLOT=0 ;;
  jp|judge+plot)                    RUN_INFERENCE=0; RUN_JUDGE=1; RUN_PLOT=1 ;;
  *)
    echo "ERROR: invalid PHASE='${PHASE}'. Allowed: all | inference | judge | plot | ij | jp" >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
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
# Completion detectors.
# With --num-runs N, inference creates run_0/ ... run_{N-1}/ subdirs.
# We check for the last run's responses to confirm inference is fully done.
# ---------------------------------------------------------------------------
_inference_complete() {
  local step=$1 dir="${OUT_DIR}/step_${step}" t last_run=$((NUM_RUNS - 1))
  local IFS=','
  for t in ${INFERENCE_TASKS}; do
    case "${t}" in
      writingbench)
        [ -e "${dir}/${t}/responses/responses.jsonl" ] || return 1 ;;
      arena-hard|alpaca-eval)
        [ -e "${dir}/${t}/model_answer" ] || return 1 ;;
      *)
        if (( NUM_RUNS > 1 )); then
          [ -e "${dir}/run_${last_run}/responses.jsonl" ] || return 1
        else
          [ -e "${dir}/${t}/responses.jsonl" ] || [ -e "${dir}/responses.jsonl" ] || return 1
        fi
        ;;
    esac
  done
  return 0
}

_judge_complete() {
  local step=$1 dir="${OUT_DIR}/step_${step}" t
  local IFS=','
  for t in ${JUDGE_TASKS}; do
    [ -f "${dir}/${t}/summary.json" ] || [ -f "${dir}/summary.json" ] || return 1
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

# ---------------------------------------------------------------------------
# Resolve per-phase step lists
# ---------------------------------------------------------------------------
declare -a PHASE1_STEPS=("${STEPS[@]}")
declare -a PHASE2_STEPS=("${STEPS[@]}")
if [ "${SKIP_COMPLETE}" = "1" ]; then
  [ "${RUN_INFERENCE}" -eq 1 ] && _filter_steps _inference_complete "infer" PHASE1_STEPS "${STEPS[@]}"
  [ "${RUN_JUDGE}"     -eq 1 ] && _filter_steps _judge_complete     "judge" PHASE2_STEPS "${STEPS[@]}"
fi

# ---------------------------------------------------------------------------
# Print effective config
# ---------------------------------------------------------------------------
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
printf "  %-20s %s\n" "NUM_RUNS"         "${NUM_RUNS} (averaged)"
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
# Helper: serve one checkpoint on one GPU, run inference (all NUM_RUNS), kill vLLM
# ---------------------------------------------------------------------------
serve_and_eval() {
  local gpu_list=$1 port=$2 model_path=$3 name=$4 out_dir=$5 tasks=$6

  echo "[GPU ${gpu_list}] Serve ${name} → :${port} | tasks: ${tasks} | runs: ${NUM_RUNS}"

  CUDA_VISIBLE_DEVICES=${gpu_list} vllm serve "${model_path}" \
    --served-model-name "${name}" \
    --host 0.0.0.0 --port "${port}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    > "${LOG_DIR}/vllm_${name}.log" 2>&1 &
  local pid=$!

  # Health check with timeout (5 min)
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
  echo "[GPU ${gpu_list}] Ready (${waited}s). Running ${NUM_RUNS} inference pass(es)..."

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
# Phase 1: Parallel inference (multi-round scheduling)
# ---------------------------------------------------------------------------
if [ "${RUN_INFERENCE}" -eq 1 ]; then
  if [ ${#PHASE1_STEPS[@]} -eq 0 ]; then
    echo "Phase 1: nothing to do (all ${#STEPS[@]} step(s) already have inference artifacts)."
  else

  num_gpus=${#GPU_IDS[@]}
  if (( num_gpus % TP_SIZE != 0 )); then
    echo "ERROR: ${num_gpus} GPUs in GPU_IDS not divisible by TP_SIZE=${TP_SIZE}" >&2
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
  echo "║  Phase 1: Inference (${#PHASE1_STEPS[@]} checkpoints, ${num_gpus} GPUs, TP=${TP_SIZE}, slots=${slots}, runs=${NUM_RUNS})"
  echo "║  GPU_IDS: ${GPU_IDS[*]}  →  slots: ${SLOT_GPUS[*]}"
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
# Phase 2: Judge-only scoring (averages across NUM_RUNS runs)
# ---------------------------------------------------------------------------
if [ "${RUN_JUDGE}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 2: Judge-only scoring (judge=${JUDGE_MODEL}, runs=${NUM_RUNS})"
  echo "╚══════════════════════════════════════════════════════════════════╝"

  if [ -z "${JUDGE_API_KEY}" ]; then
    echo "WARNING: JUDGE_API_KEY is empty. Set it in .env or in the CONFIG section."
    echo "Skipping judge phase."
  else
    if [ ${#PHASE2_STEPS[@]} -eq 0 ]; then
      echo "Phase 2: nothing to do (all ${#STEPS[@]} step(s) already judged)."
    else
      judge_one() {
        local step=$1 name="s${1}"
        local dir="${OUT_DIR}/step_${step}"
        echo "[step_${step}] Judging ${NUM_RUNS} run(s) (${JUDGE_TASKS})..."
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
        echo "[step_${step}] Judge done. Averaged summary: ${dir}/summary.json"
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
# Phase 3: Plot training curves
# ---------------------------------------------------------------------------
if [ "${RUN_PLOT}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 3: Plotting                                             "
  echo "╚══════════════════════════════════════════════════════════════════╝"

  steps_csv=$(IFS=,; echo "${STEPS[*]}")

  echo "  Aggregating ${NUM_RUNS} run(s) per checkpoint into summary_agg.json..."
  python "${EVAL_FRAMEWORK_ROOT}/tools/aggregate_runs.py" \
    --out-dir "${OUT_DIR}" \
    --steps "${steps_csv}" \
    --tasks "healthbench" \
    --n-samples "${NUM_RUNS}"

  python "${EVAL_FRAMEWORK_ROOT}/tools/plot_training_curves.py" \
    --runs "${EXP_LABEL}=${OUT_DIR}" \
    --name-pattern "${EXP_LABEL}=step_{step}" \
    --steps "${steps_csv}" \
    --tasks "healthbench" \
    --plot-dir "${PLOT_DIR}"

  echo ""
  echo "Plots saved to: ${PLOT_DIR}/"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  All done!                                                      "
echo "║  Outputs : ${OUT_DIR}/                               "
echo "║  Plots   : ${PLOT_DIR}/                              "
echo "║  Logs    : ${LOG_DIR}/                               "
echo "╚══════════════════════════════════════════════════════════════════╝"
