#!/usr/bin/env bash
# Poll WAM /health until workers_loaded=true (pool finished eager startup).
set -euo pipefail

WAM_HEALTH_URL="${WAM_HEALTH_URL:-http://127.0.0.1:${WAM_PORT:-8002}/health}"
WAM_READY_TIMEOUT_SEC="${WAM_READY_TIMEOUT_SEC:-1200}"
WAM_READY_POLL_SEC="${WAM_READY_POLL_SEC:-5}"

_wam_poll_status() {
  "${KAIROS_WAM_PYTHON:-python3}" -c '
import json, sys
d = json.loads(sys.argv[1])
if d.get("workers_loaded"):
    print("ready")
elif d.get("load_failed") or d.get("load_error"):
    err = (d.get("load_error") or "worker pool load failed").strip()
    print("failed:" + err)
elif d.get("workers_loading"):
    print("loading")
else:
    print("waiting")
' "$1"
}

start_ts=$(date +%s)
echo "[wait_wam] url=${WAM_HEALTH_URL} timeout=${WAM_READY_TIMEOUT_SEC}s poll=${WAM_READY_POLL_SEC}s"
echo "[wait_wam] loading 4 workers typically takes 3-5 minutes; eval starts after workers_loaded=true"

while true; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  if (( elapsed >= WAM_READY_TIMEOUT_SEC )); then
    echo "[wait_wam] ERROR: timeout after ${WAM_READY_TIMEOUT_SEC}s" >&2
    exit 1
  fi

  if resp=$(curl -sf --max-time 10 "${WAM_HEALTH_URL}" 2>/dev/null); then
    status=$(_wam_poll_status "${resp}")
    case "${status}" in
      ready)
        echo "[wait_wam] ready (${elapsed}s): ${resp}"
        exit 0
        ;;
      loading)
        echo "[wait_wam] (${elapsed}s) WAM up, loading models"
        ;;
      waiting)
        echo "[wait_wam] (${elapsed}s) WAM up, waiting"
        ;;
      failed:*)
        err="${status#failed:}"
        echo "[wait_wam] ERROR: WAM load failed: ${err}" >&2
        echo "[wait_wam] last /health: ${resp}" >&2
        exit 1
        ;;
      *)
        echo "[wait_wam] ERROR: unexpected status: ${status}" >&2
        exit 1
        ;;
    esac
  else
    echo "[wait_wam] (${elapsed}s) cannot reach ${WAM_HEALTH_URL} (server starting?)"
  fi

  sleep "${WAM_READY_POLL_SEC}"
done
