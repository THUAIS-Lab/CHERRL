#!/bin/bash
echo "[INFO] Stopping all vLLM and load balancer processes..."
if [[ ! -f "/WORK/PUBLIC/lijuanzi_work/XuekangWang/CHERRL/vllm_logs/pids.txt" ]]; then
    echo "[WARN] PID file not found: /WORK/PUBLIC/lijuanzi_work/XuekangWang/CHERRL/vllm_logs/pids.txt"
    exit 1
fi
while IFS= read -r pid; do
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && echo "[INFO] Killed PID $pid"
    else
        echo "[INFO] PID $pid already stopped"
    fi
done < "/WORK/PUBLIC/lijuanzi_work/XuekangWang/CHERRL/vllm_logs/pids.txt"
rm -f "/WORK/PUBLIC/lijuanzi_work/XuekangWang/CHERRL/vllm_logs/pids.txt"
echo "[INFO] Done."
