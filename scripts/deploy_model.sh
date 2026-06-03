#!/bin/bash
set -euo pipefail

# ============================================================
# deploy_model.sh
# Deploy vLLM model on multiple GPUs with load balancing.
# Processes are daemonized (nohup+disown) so they survive
# SSH disconnection. A stop script is generated on startup.
# ============================================================

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --model-name NAME     Model name served on the API
  --model-path PATH     Path to model weights
  --gpus IDS            Comma-separated GPU IDs, e.g. "0,1,2,3"

Optional:
  --base-port PORT      First backend port (default: 8001)
  --lb-port   PORT      Unified load-balancer port (default: 8000)
  --extra-args ARGS     Extra args forwarded to vllm serve
  --log-dir   DIR       Log directory (default: ./vllm_logs)
  -h, --help

Example:
  $0 --model-name Qwen3-4B --model-path /models/Qwen3-4B --gpus 0,1,2,3
  $0 --model-name Qwen3-4B --model-path /models/Qwen3-4B --gpus 0,1 --extra-args "--max-model-len 8192"
EOF
    exit 1
}

# ---------- defaults ----------
MODEL_NAME=""
MODEL_PATH=""
GPUS=""
BASE_PORT=8001
LB_PORT=8000
EXTRA_ARGS=""
LOG_DIR="./vllm_logs"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --gpus)       GPUS="$2";       shift 2 ;;
        --base-port)  BASE_PORT="$2";  shift 2 ;;
        --lb-port)    LB_PORT="$2";    shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        --log-dir)    LOG_DIR="$2";    shift 2 ;;
        -h|--help)    usage ;;
        *) echo "[ERROR] Unknown option: $1"; usage ;;
    esac
done

[[ -z "$MODEL_NAME" ]] && { echo "[ERROR] --model-name is required"; usage; }
[[ -z "$MODEL_PATH" ]] && { echo "[ERROR] --model-path is required"; usage; }
[[ -z "$GPUS"       ]] && { echo "[ERROR] --gpus is required"; usage; }

mkdir -p "$LOG_DIR"
LOG_DIR="$(realpath "$LOG_DIR")"  # use absolute path so stop script works from anywhere

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

PID_FILE="$LOG_DIR/pids.txt"
STOP_SCRIPT="$LOG_DIR/stop_model.sh"
LB_SCRIPT="$LOG_DIR/lb_server.py"

# ---------- write the load-balancer script ----------
cat > "$LB_SCRIPT" <<'PYEOF'
#!/usr/bin/env python3
"""
Least-connections async reverse proxy for vLLM backends.
Optimized for concurrent (non-streaming) API calls.
"""
import argparse
import logging
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LB] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lb")

class Backend:
    def __init__(self, url: str):
        self.url = url
        self.active = 0

backends: list[Backend] = []

def pick_backend() -> Backend:
    return min(backends, key=lambda b: b.active)

async def proxy(request: web.Request) -> web.Response:
    backend = pick_backend()
    backend.active += 1
    target = backend.url + str(request.rel_url)
    try:
        body = await request.read()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        async with request.app["session"].request(
            method=request.method,
            url=target,
            headers=headers,
            data=body,
        ) as resp:
            resp_body = await resp.read()
            return web.Response(
                status=resp.status,
                headers={
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ("transfer-encoding", "content-encoding")
                },
                body=resp_body,
            )
    except Exception as e:
        log.error("Backend %s error: %s", backend.url, e)
        return web.Response(status=502, text=f"Bad gateway: {e}")
    finally:
        backend.active -= 1

async def health(request: web.Request) -> web.Response:
    info = [{"backend": b.url, "active_requests": b.active} for b in backends]
    return web.json_response({"status": "ok", "backends": info})

async def on_startup(app: web.Application):
    connector = TCPConnector(limit=0, keepalive_timeout=30)
    timeout = ClientTimeout(total=300, connect=10)
    app["session"] = ClientSession(connector=connector, timeout=timeout)
    log.info("Load balancer ready. Backends: %s", [b.url for b in backends])

async def on_shutdown(app: web.Application):
    await app["session"].close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    parser.add_argument("--lb-port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    for port in args.ports:
        backends.append(Backend(f"http://127.0.0.1:{port}"))

    app = web.Application()
    app.router.add_route("*", "/health_lb", health)
    app.router.add_route("*", "/{path_info:.*}", proxy)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.host, port=args.lb_port, access_log=None)

if __name__ == "__main__":
    main()
PYEOF

# ---------- start vLLM per GPU (daemonized) ----------
PORTS=()
ALL_PIDS=()

for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    PORT=$((BASE_PORT + i))
    PORTS+=("$PORT")
    LOG_FILE="$LOG_DIR/vllm_gpu${GPU_ID}_port${PORT}.log"

    echo "[INFO] Starting vLLM on GPU $GPU_ID → port $PORT (log: $LOG_FILE)"

    nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
        --model '${MODEL_PATH}' \
        --served-model-name '${MODEL_NAME}' \
        --port ${PORT} \
        --host 127.0.0.1 \
        ${EXTRA_ARGS}" \
        > "$LOG_FILE" 2>&1 &

    disown $!
    ALL_PIDS+=($!)
done

# ---------- wait for all backends to become healthy ----------
echo "[INFO] Waiting for $NUM_GPUS backend(s) to become healthy..."
HEALTH_TIMEOUT=300
for i in "${!GPU_ARRAY[@]}"; do
    PORT="${PORTS[$i]}"
    GPU_ID="${GPU_ARRAY[$i]}"
    DEADLINE=$((SECONDS + HEALTH_TIMEOUT))
    printf "[INFO] Polling GPU %s port %d " "$GPU_ID" "$PORT"
    while true; do
        if curl -sf "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null; then
            echo " ready."
            break
        fi
        if [[ $SECONDS -ge $DEADLINE ]]; then
            echo ""
            echo "[ERROR] GPU $GPU_ID (port $PORT) did not become healthy within ${HEALTH_TIMEOUT}s."
            echo "        Check log: $LOG_DIR/vllm_gpu${GPU_ID}_port${PORT}.log"
            exit 1
        fi
        printf "."
        sleep 3
    done
done

# ---------- start load balancer (daemonized) ----------
echo "[INFO] Starting load balancer on port $LB_PORT..."
LB_LOG="$LOG_DIR/lb.log"

nohup python3 "$LB_SCRIPT" --ports "${PORTS[@]}" --lb-port "$LB_PORT" \
    > "$LB_LOG" 2>&1 &
disown $!
LB_PID=$!
ALL_PIDS+=($LB_PID)

sleep 2
if ! kill -0 "$LB_PID" 2>/dev/null; then
    echo "[ERROR] Load balancer failed to start. Check: $LB_LOG"
    exit 1
fi

# ---------- write PID file ----------
printf "%s\n" "${ALL_PIDS[@]}" > "$PID_FILE"

# ---------- write stop script ----------
cat > "$STOP_SCRIPT" <<STOPEOF
#!/bin/bash
echo "[INFO] Stopping all vLLM and load balancer processes..."
if [[ ! -f "$PID_FILE" ]]; then
    echo "[WARN] PID file not found: $PID_FILE"
    exit 1
fi
while IFS= read -r pid; do
    if kill -0 "\$pid" 2>/dev/null; then
        kill "\$pid" && echo "[INFO] Killed PID \$pid"
    else
        echo "[INFO] PID \$pid already stopped"
    fi
done < "$PID_FILE"
rm -f "$PID_FILE"
echo "[INFO] Done."
STOPEOF
chmod +x "$STOP_SCRIPT"

# ---------- summary ----------
echo ""
echo "========================================"
echo "  Model     : $MODEL_NAME"
echo "  GPUs      : $GPUS"
echo "  Backends  : ${PORTS[*]} (127.0.0.1)"
echo "  API URL   : http://0.0.0.0:$LB_PORT/v1"
echo "  Health    : http://0.0.0.0:$LB_PORT/health_lb"
echo "  Logs      : $LOG_DIR/"
echo "  Stop with : bash $STOP_SCRIPT"
echo "========================================"
