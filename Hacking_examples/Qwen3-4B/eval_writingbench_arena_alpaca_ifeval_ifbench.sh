#!/usr/bin/env bash
# =============================================================================
# eval_writingbench_arena_alpaca_ifeval_ifbench.sh
#
# Evaluates Qwen3-4B_healthbench_tone_bias checkpoints on HealthBench,
# WritingBench, Arena-Hard, AlpacaEval, IFEval, and IFBench.
# All six benchmarks are run and plotted in one pipeline.
#
# IFEval / IFBench notes:
#   * Both are rule-based instruction-following benchmarks and DO NOT need a
#     judge model. They complete fully during the inference phase (responses
#     are generated against the local vLLM server, then metrics are computed
#     in-process via NLTK + IFBench's checker library).
#   * IFBench requires Python deps `emoji` and `syllapy` and a clone of the
#     IFBench checker source. We point at the bundled copy under
#     evaluation/IF-EVAL by default — override with IFBENCH_DIR if needed.
#   * NLTK resources are downloaded on first run unless you set
#     IFEVAL_SKIP_NLTK_DOWNLOAD=1 / IFBENCH_SKIP_NLTK_DOWNLOAD=1.
#
# Full pipeline:  Inference (vLLM) → Judge (API, judge-needing tasks only)
#                 → Plot (all 6 tasks)
#
# Usage:
#   bash eval_writingbench_arena_alpaca_ifeval_ifbench.sh
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

# Steps to evaluate
: "${STEPS:=0}"

# All tasks. ifeval/ifbench finish end-to-end during the inference phase (no
# judge model required), so they only appear in INFERENCE_TASKS.
# healthbench benefits from multiple independent runs; use HEALTHBENCH_NUM_RUNS
# to override (default 3). Other tasks only need 1 run.
: "${INFERENCE_TASKS:=healthbench,writingbench,arena-hard,alpaca-eval,ifeval,ifbench}"
: "${JUDGE_TASKS:=healthbench,writingbench,arena-hard,alpaca-eval}"

: "${HEALTHBENCH_NUM_RUNS:=3}"

# These tasks don't benefit from multiple independent runs
: "${NUM_RUNS:=1}"

# GPU & vLLM settings
: "${GPU_IDS:=6}"
: "${TP_SIZE:=1}"
: "${GPU_MEM_UTIL:=0.85}"
: "${BASE_PORT:=30001}"
: "${INFERENCE_THREADS:=512}"

# Judge settings (only used by writingbench/arena-hard/alpaca-eval)
: "${JUDGE_MODEL:=${AGENT_MODEL:-qwen-plus}}"
: "${JUDGE_BASE_URL:=${AGENT_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
: "${JUDGE_API_KEY:=${AGENT_API_KEY:-}}"
: "${JUDGE_THREADS:=32}"
: "${JUDGE_BATCH_SIZE:=5}"

# IFEval / IFBench specifics
: "${IFBENCH_DIR:=/data/nvme1/wangxk/hackingRubricsRL/evaluation/IFBench}"
: "${IFEVAL_SKIP_NLTK_DOWNLOAD:=0}"
: "${IFBENCH_SKIP_NLTK_DOWNLOAD:=0}"

: "${LOG_DIR:=/data/nvme1/wangxk/eval_logs/${EXP_LABEL}_6tasks}"

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

# Build extra-flags arrays used by both phase1 and phase2 invocations.
INFER_EXTRA_ARGS=()
[ -n "${IFBENCH_DIR}" ]           && INFER_EXTRA_ARGS+=(--ifbench-dir "${IFBENCH_DIR}")
[ "${IFEVAL_SKIP_NLTK_DOWNLOAD}"  = "1" ] && INFER_EXTRA_ARGS+=(--ifeval-skip-nltk-download)
[ "${IFBENCH_SKIP_NLTK_DOWNLOAD}" = "1" ] && INFER_EXTRA_ARGS+=(--ifbench-skip-nltk-download)

# ---------------------------------------------------------------------------
# Completion detectors
# ---------------------------------------------------------------------------
_inference_complete() {
  local step=$1 dir="${OUT_DIR}/step_${step}" t
  local IFS=','
  for t in ${INFERENCE_TASKS}; do
    case "${t}" in
      healthbench)
        if (( HEALTHBENCH_NUM_RUNS > 1 )); then
          [ -e "${dir}/run_$((HEALTHBENCH_NUM_RUNS-1))/responses.jsonl" ] || return 1
        else
          [ -e "${dir}/${t}/responses.jsonl" ] || return 1
        fi
        ;;
      writingbench)
        [ -e "${dir}/${t}/responses/responses.jsonl" ] || return 1 ;;
      arena-hard|alpaca-eval)
        [ -e "${dir}/${t}/model_answer" ] || return 1 ;;
      ifeval|ifbench)
        [ -e "${dir}/${t}/summary.json" ] || return 1 ;;
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
printf "  %-22s %s\n" "EXP_LABEL"        "${EXP_LABEL}"
printf "  %-22s %s\n" "CKPT_DIR"         "${CKPT_DIR}"
printf "  %-22s %s\n" "OUT_DIR"          "${OUT_DIR}"
printf "  %-22s %s\n" "PLOT_DIR"         "${PLOT_DIR}"
printf "  %-22s %s\n" "LOG_DIR"          "${LOG_DIR}"
printf "  %-22s %s (%d step(s))\n" "STEPS"  "${STEPS[*]}" "${#STEPS[@]}"
if [ "${SKIP_COMPLETE}" = "1" ]; then
  [ "${RUN_INFERENCE}" -eq 1 ] && \
    printf "  %-22s %s (%d step(s))\n" "  → phase1 inference" "${PHASE1_STEPS[*]:-<none>}" "${#PHASE1_STEPS[@]}"
  [ "${RUN_JUDGE}" -eq 1 ] && \
    printf "  %-22s %s (%d step(s))\n" "  → phase2 judge"     "${PHASE2_STEPS[*]:-<none>}" "${#PHASE2_STEPS[@]}"
fi
printf "  %-22s %s (%d gpu(s), TP=%s)\n" "GPU_IDS" "${GPU_IDS[*]}" "${#GPU_IDS[@]}" "${TP_SIZE}"
printf "  %-22s %s (healthbench: %s)\n" "NUM_RUNS" "${NUM_RUNS}" "${HEALTHBENCH_NUM_RUNS}"
printf "  %-22s %s\n" "INFERENCE_TASKS"  "${INFERENCE_TASKS}"
printf "  %-22s %s\n" "JUDGE_TASKS"      "${JUDGE_TASKS}"
printf "  %-22s %s\n" "JUDGE_MODEL"      "${JUDGE_MODEL}"
printf "  %-22s %s\n" "JUDGE_BASE_URL"   "${JUDGE_BASE_URL}"
printf "  %-22s %s\n" "JUDGE_API_KEY"    "$(_mask "${JUDGE_API_KEY}")"
printf "  %-22s %s\n" "IFBENCH_DIR"      "${IFBENCH_DIR}"
printf "  %-22s %s / %s\n" "SKIP_NLTK (ifeval/ifbench)" "${IFEVAL_SKIP_NLTK_DOWNLOAD}" "${IFBENCH_SKIP_NLTK_DOWNLOAD}"
printf "  %-22s %s  (inference=%s judge=%s plot=%s)\n" "PHASE" "${PHASE}" "${RUN_INFERENCE}" "${RUN_JUDGE}" "${RUN_PLOT}"
printf "  %-22s %s\n" "SKIP_COMPLETE"    "${SKIP_COMPLETE}"
echo ""

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1 — exiting without executing any phase."
  exit 0
fi

# ---------------------------------------------------------------------------
# Helper: serve one checkpoint, run inference for all tasks, kill vLLM.
# healthbench uses HEALTHBENCH_NUM_RUNS; all other tasks use NUM_RUNS.
# ---------------------------------------------------------------------------
serve_and_eval() {
  local gpu_list=$1 port=$2 model_path=$3 name=$4 out_dir=$5

  echo "[GPU ${gpu_list}] Serve ${name} → :${port} | tasks: ${INFERENCE_TASKS}"

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

  # healthbench: run HEALTHBENCH_NUM_RUNS independent passes for reliable averaging.
  if [[ ",${INFERENCE_TASKS}," == *",healthbench,"* ]]; then
    echo "[GPU ${gpu_list}] healthbench (${HEALTHBENCH_NUM_RUNS} runs)..."
    eval-framework \
      --tasks "healthbench" \
      --model "${name}" \
      --base-url "http://localhost:${port}/v1" \
      --inference-only \
      --num-runs "${HEALTHBENCH_NUM_RUNS}" \
      --output-dir "${out_dir}" \
      --num-threads "${INFERENCE_THREADS}" \
      2>&1 | tee "${LOG_DIR}/eval_${name}.log"
  fi

  # All other tasks: single run.
  # --inference-only is honored by writingbench/arena-hard/alpaca-eval; ifeval
  # and ifbench ignore it and run their (rule-based) eval to completion.
  local other_tasks
  other_tasks=$(printf '%s' "${INFERENCE_TASKS}" | tr ',' '\n' | grep -v '^healthbench$' | paste -sd ',' -)
  if [ -n "${other_tasks}" ]; then
    echo "[GPU ${gpu_list}] ${other_tasks} (${NUM_RUNS} run(s))..."
    eval-framework \
      --tasks "${other_tasks}" \
      --model "${name}" \
      --base-url "http://localhost:${port}/v1" \
      --inference-only \
      --num-runs "${NUM_RUNS}" \
      --output-dir "${out_dir}" \
      --num-threads "${INFERENCE_THREADS}" \
      "${INFER_EXTRA_ARGS[@]}" \
      2>&1 | tee -a "${LOG_DIR}/eval_${name}.log"
  fi

  echo "[GPU ${gpu_list}] Done: ${name}"

  kill -- -${pid} 2>/dev/null || kill ${pid} 2>/dev/null || true
  wait ${pid} 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Phase 1: Parallel inference (all 5 tasks per checkpoint)
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
  echo "║  healthbench: ${HEALTHBENCH_NUM_RUNS} runs  |  others: ${NUM_RUNS} run(s)"
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

      serve_and_eval "${gpu_list}" ${port} "${model_path}" "${name}" "${out_dir}" &
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
  echo "║  healthbench: ${HEALTHBENCH_NUM_RUNS} runs  |  others: ${NUM_RUNS} run(s)"
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
        local log="${LOG_DIR}/judge_s${step}.log"

        # healthbench: judge all HEALTHBENCH_NUM_RUNS inference runs.
        if [[ ",${JUDGE_TASKS}," == *",healthbench,"* ]]; then
          echo "[step_${step}] Judging healthbench (${HEALTHBENCH_NUM_RUNS} runs)..."
          eval-framework \
            --tasks "healthbench" \
            --model "${name}" \
            --judge-model "${JUDGE_MODEL}" \
            --judge-base-url "${JUDGE_BASE_URL}" \
            --judge-api-key "${JUDGE_API_KEY}" \
            --output-dir "${dir}" \
            --judge-only \
            --num-runs "${HEALTHBENCH_NUM_RUNS}" \
            --num-threads "${JUDGE_THREADS}" \
            2>&1 | tee "${log}"
        fi

        # Other tasks: single run.
        local other_judge_tasks
        other_judge_tasks=$(printf '%s' "${JUDGE_TASKS}" | tr ',' '\n' | grep -v '^healthbench$' | paste -sd ',' -)
        if [ -n "${other_judge_tasks}" ]; then
          echo "[step_${step}] Judging ${other_judge_tasks} (${NUM_RUNS} run(s))..."
          eval-framework \
            --tasks "${other_judge_tasks}" \
            --model "${name}" \
            --judge-model "${JUDGE_MODEL}" \
            --judge-base-url "${JUDGE_BASE_URL}" \
            --judge-api-key "${JUDGE_API_KEY}" \
            --output-dir "${dir}" \
            --judge-only \
            --num-runs "${NUM_RUNS}" \
            --num-threads "${JUDGE_THREADS}" \
            2>&1 | tee -a "${log}"
        fi

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
# Phase 3: Aggregate healthbench runs + plot all 6 benchmarks
# ---------------------------------------------------------------------------
if [ "${RUN_PLOT}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 3: Plotting (healthbench + writingbench + arena-hard"
  echo "║           + alpaca-eval + ifeval + ifbench)"
  echo "╚══════════════════════════════════════════════════════════════════╝"

  steps_csv=$(IFS=,; echo "${STEPS[*]}")

  if (( HEALTHBENCH_NUM_RUNS > 1 )); then
    echo "  Aggregating ${HEALTHBENCH_NUM_RUNS} healthbench run(s) per checkpoint..."
    python "${EVAL_FRAMEWORK_ROOT}/tools/aggregate_runs.py" \
      --out-dir "${OUT_DIR}" \
      --steps "${steps_csv}" \
      --tasks "healthbench" \
      --n-samples "${HEALTHBENCH_NUM_RUNS}"
  fi

  python "${EVAL_FRAMEWORK_ROOT}/tools/plot_training_curves.py" \
    --runs "${EXP_LABEL}=${OUT_DIR}" \
    --name-pattern "${EXP_LABEL}=step_{step}" \
    --steps "${steps_csv}" \
    --tasks "healthbench,writingbench,arena-hard,alpaca-eval,ifeval,ifbench" \
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
