#!/usr/bin/env bash
# Budget-ablation launcher (canonical) — RHDA across budgets {5,10,20,30,40,50} × 3 reps.
#
# Usage:
#   bash detection/agent_compare/budget_ablation/launch.sh \
#       --run <run_id> \
#       --mirror-dir <path/to/sanitized/mirror> \
#       --output-dir <path/to/per-rep-output> \
#       [--wrapper <path/to/paper-prompt-wrapper>] \
#       [--prompt-version paper_final]
#
# Required:
#   --run               run identifier (any string; used as a label in progress logs)
#   --mirror-dir        path to the sanitized 4-field rollout mirror for the run
#   --output-dir        where per-budget per-rep agent_workspace output will be written
#
# Dispatch:
#   - If --wrapper is provided, the launcher invokes the wrapper script
#     with the same argument set as `python -m detection.rhda`. Use this
#     for paper-final rerun configurations that ship their own prompt
#     wrapper (e.g. the run_E paper-prompt wrapper from the external release).
#   - Otherwise the launcher invokes `python -m detection.rhda` directly,
#     with --prompt-version kept as a compatibility label. The release ships
#     a single canonical RHDA prompt at `detection/rhda/prompts.py`.
#
# Optional env:
#   PY                  python interpreter (default: python3)
#   API_MODEL           OpenAI-compatible model id (default: qwen3.5-plus)
#   MAX_LOOP_ITERATIONS (default: 120)
#   TEMPERATURE         (default: 0.0)
#   HTTPS_PROXY         outbound proxy if your env requires one
#
# This launcher does NOT bake in user paths or proxy hosts. It only
# orchestrates the 18-rep sweep (6 budgets × 3 reps) calling either the
# wrapper or the canonical RHDA entry, with sentinel-based resumability.

set -u
set -o pipefail

usage() {
  sed -n '1,/^set -u/p' "$0" | sed 's/^# \?//; /^set -u/d'
  exit "${1:-0}"
}

RUN=""
MIRROR_DIR=""
OUTPUT_DIR=""
WRAPPER=""
PROMPT_VERSION="paper_final"
PY="${PY:-python3}"
API_MODEL="${API_MODEL:-qwen3.5-plus}"
MAX_LOOP_ITERATIONS="${MAX_LOOP_ITERATIONS:-120}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BUDGETS_DEFAULT=(5 10 20 30 40 50)
REPS_DEFAULT=(1 2 3)

while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN="$2"; shift 2;;
    --mirror-dir) MIRROR_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --wrapper) WRAPPER="$2"; shift 2;;
    --prompt-version) PROMPT_VERSION="$2"; shift 2;;
    -h|--help) usage 0;;
    *) echo "unknown arg: $1"; usage 2;;
  esac
done

[ -z "$RUN" ] && { echo "missing --run"; usage 2; }
[ -z "$MIRROR_DIR" ] && { echo "missing --mirror-dir"; usage 2; }
[ -z "$OUTPUT_DIR" ] && { echo "missing --output-dir"; usage 2; }

if [ -n "$WRAPPER" ]; then
  [ -f "$WRAPPER" ] || { echo "wrapper not found: $WRAPPER"; exit 2; }
  unset DETECTION_AGENT_PROMPT_VERSION
else
  export DETECTION_AGENT_PROMPT_VERSION="$PROMPT_VERSION"
fi

mkdir -p "$OUTPUT_DIR"
PROGRESS="$OUTPUT_DIR/_progress_budget_${RUN}.log"
echo "[$(date -Iseconds)] launcher started (${RUN} budget ablation)" >> "$PROGRESS"

run_budget() {
  local budget=$1
  local rep=$2
  local REP_DIR="$OUTPUT_DIR/budget_${budget}/rep${rep}"
  local SENT="$REP_DIR/_wall_seconds.txt"
  mkdir -p "$REP_DIR"
  if [ -f "$SENT" ]; then
    echo "[$(date -Iseconds)] skip budget=${budget} rep${rep} (done)" >> "$PROGRESS"
    return
  fi
  local LOG="$REP_DIR/_run.log"
  echo "[$(date -Iseconds)] START budget=${budget} rep${rep}" >> "$PROGRESS"
  local START=$SECONDS
  if [ -n "$WRAPPER" ]; then
    "$PY" "$WRAPPER" \
      --model "$API_MODEL" \
      --rollout-dir "$MIRROR_DIR" --output-dir "$REP_DIR" \
      --max-tool-calls "$budget" \
      --max-loop-iterations "$MAX_LOOP_ITERATIONS" \
      --temperature "$TEMPERATURE" > "$LOG" 2>&1
  else
    "$PY" -m detection.rhda \
      --model "$API_MODEL" \
      --rollout-dir "$MIRROR_DIR" --output-dir "$REP_DIR" \
      --max-tool-calls "$budget" \
      --max-loop-iterations "$MAX_LOOP_ITERATIONS" \
      --temperature "$TEMPERATURE" > "$LOG" 2>&1
  fi
  local RC=$?
  local DUR=$((SECONDS - START))
  if [ "$RC" -eq 0 ]; then
    echo "$DUR" > "$SENT"
    echo "[$(date -Iseconds)] DONE  budget=${budget} rep${rep} dur=${DUR}s" >> "$PROGRESS"
  else
    echo "[$(date -Iseconds)] FAIL  budget=${budget} rep${rep} dur=${DUR}s rc=$RC" >> "$PROGRESS"
  fi
}

for B in "${BUDGETS_DEFAULT[@]}"; do
  for rep in "${REPS_DEFAULT[@]}"; do
    run_budget "$B" "$rep"
  done
  echo "[$(date -Iseconds)] FINISHED budget=${B}" >> "$PROGRESS"
done

echo "[$(date -Iseconds)] launcher finished (${RUN}, all 6 budgets × 3 reps)" >> "$PROGRESS"
