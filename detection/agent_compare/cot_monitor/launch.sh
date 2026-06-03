#!/usr/bin/env bash
# CoT monitor launcher (canonical) — wraps the step-wise CoT no-score monitor.
#
# Usage:
#   bash detection/agent_compare/cot_monitor/launch.sh --run <run_id> --rep 1 \
#       [--mirror-dir <CoT-no-score-mirror>] [--output-dir <out>] [--monitor stepwise|stateful] [--prompt-file <path>]
#
# Required:
#   --run         run identifier (any string; used as a directory label)
#   --rep         integer (1, 2, 3, ...)
#
# Optional (auto-defaults):
#   --monitor     stepwise (default) | stateful
#   --mirror-dir  CoT no-score mirror dir (default: detection/datasets/cot_noscore/<run_id>/)
#   --output-dir  where per-rep results land (default: $DATA_ROOT/agent_compare/cot_monitor/<run>/rep<rep>)
#   --prompt-file path to monitor prompt md (default: detection/agent_compare/cot_monitor/prompt[_stepwise].md)
#
# This launcher does NOT hard-code paths. Required external data (CoT no-score
# mirror) lives in the external release; see detection/docs/RESTORE_DATA.md.

set -u
set -o pipefail

usage() {
  sed -n '1,/^set -u/p' "$0" | sed 's/^# \?//; /^set -u/d'
  exit "${1:-0}"
}

RUN=""
REP=""
MONITOR="stepwise"
MIRROR_DIR=""
OUTPUT_DIR=""
PROMPT_FILE=""
PY="${PY:-python3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN="$2"; shift 2;;
    --rep) REP="$2"; shift 2;;
    --monitor) MONITOR="$2"; shift 2;;
    --mirror-dir) MIRROR_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --prompt-file) PROMPT_FILE="$2"; shift 2;;
    -h|--help) usage 0;;
    *) echo "unknown arg: $1"; usage 2;;
  esac
done

[ -z "$RUN" ] && { echo "missing --run"; usage 2; }
[ -z "$REP" ] && { echo "missing --rep"; usage 2; }

case "$MONITOR" in
  stepwise|stateful) ;;
  *) echo "unsupported --monitor: $MONITOR (use stepwise | stateful)"; exit 2;;
esac

# Dispatch to the runner module; --run / --rep are forwarded.
RUNNER_MOD="detection.agent_compare.cot_monitor.stepwise_runner"
if [ "$MONITOR" = "stateful" ]; then
  RUNNER_MOD="detection.agent_compare.cot_monitor.runner"
fi

ARGS=(--run "$RUN" --rep "$REP")
[ -n "$MIRROR_DIR"   ] && ARGS+=(--mirror-dir "$MIRROR_DIR")
[ -n "$OUTPUT_DIR"   ] && ARGS+=(--output-dir "$OUTPUT_DIR")
[ -n "$PROMPT_FILE"  ] && ARGS+=(--prompt-file "$PROMPT_FILE")

START=$SECONDS
"$PY" -m "$RUNNER_MOD" "${ARGS[@]}"
RC=$?
DUR=$((SECONDS - START))
echo "[$(date -Iseconds)] CoT ${MONITOR} monitor ${RUN} rep${REP} dur=${DUR}s rc=$RC"
exit "$RC"
