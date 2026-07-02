#!/usr/bin/env bash
# Start the WAM inference server, wait for it, then launch the RoboTwin eval client.
# Usage:
#   bash run.sh
# Key env vars:
#   WAM_PRETRAINED_DIT            - path to WAM checkpoint (.safetensors)
#   WAM_GPU_IDS                   - GPU IDs for WAM server (default: 0,1,2,3)
#   DATASET_STATS_PATH            - path to dataset_stats.json
#   EVAL_OUTPUT_DIR               - eval output directory
#   EVAL_GPU_IDS                  - GPU IDs for eval client workers (default: WAM_GPU_IDS)
#   ROBOTWIN_ROOT                 - path to RoboTwin repo (default: third_party/RoboTwin)

set -euo pipefail
export HYDRA_FULL_ERROR=1

_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${_project_dir}}"
export COMMON_ROOT="${COMMON_ROOT:-$(cd "${PROJECT_ROOT}/../common" && pwd)}"
export KAIROS_WAM_ROOT="${KAIROS_WAM_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${PROJECT_ROOT}/third_party/RoboTwin}"
export EVAL_STEP_LIMIT_FILE="${EVAL_STEP_LIMIT_FILE:-${PROJECT_ROOT}/robotwin_eval/config/_eval_step_limit.yml}"
export WAM_CFG_PATH="${WAM_CFG_PATH:-${PROJECT_ROOT}/configs/robotwin_wam_infer_config.py}"
export WAM_HOST="${WAM_HOST:-0.0.0.0}"
export WAM_PORT="${WAM_PORT:-8006}"
export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:${WAM_PORT}}"
export PYTHONPATH="${KAIROS_WAM_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS:-ATEN,TRITON,CPP}"
export WAM_GPU_IDS="${WAM_GPU_IDS:-0,1,2,3}"
export EVAL_GPU_IDS="${EVAL_GPU_IDS:-${WAM_GPU_IDS}}"

export KAIROS_MODEL_DIR="${KAIROS_MODEL_DIR:-/path/to/share_models/git_models}"
export WAM_EAGER_LOAD_ON_STARTUP="${WAM_EAGER_LOAD_ON_STARTUP:-1}"
export WAM_SKIP_LOAD_ENGINE="${WAM_SKIP_LOAD_ENGINE:-1}"
export WAM_PRETRAINED_DIT="${WAM_PRETRAINED_DIT:-/path/to/step-XXXXX.safetensors}"
_default_output_root="${PROJECT_ROOT}/../../outputs/robotwin"
mkdir -p "${_default_output_root}"
_default_output_root="$(cd "${_default_output_root}" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${_default_output_root}}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/eval}"

export DATASET_STATS_PATH="${DATASET_STATS_PATH:-${PROJECT_ROOT}/robotwin_dataset_stats.json}"

export SERVER_PYTHON="${SERVER_PYTHON:-/path/to/kairos-wam/bin/python}"
export KAIROS_WAM_PYTHON="${KAIROS_WAM_PYTHON:-${SERVER_PYTHON}}"
export CLIENT_PYTHON="${CLIENT_PYTHON:-/path/to/robotwin-eval/bin/python}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${OUTPUT_ROOT}/torch_inductor_cache}"

WAM_LOG="${WAM_LOG:-${OUTPUT_ROOT}/wam_server.log}"
mkdir -p "$(dirname "${WAM_LOG}")" "${OUTPUT_ROOT}" "${EVAL_OUTPUT_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

echo "[run] server_python=${SERVER_PYTHON}"
echo "[run] client_python=${CLIENT_PYTHON}"
echo "[run] output_root=${OUTPUT_ROOT}"
echo "[run] eval_output_dir=${EVAL_OUTPUT_DIR}"
echo "[run] dataset_stats_path=${DATASET_STATS_PATH}"
echo "[run] wam_gpus=${WAM_GPU_IDS} eval_gpu_ids=${EVAL_GPU_IDS} endpoint=${MODEL_ENDPOINT}"

(
  cd "${COMMON_ROOT}"
  exec "${SERVER_PYTHON}" -m uvicorn wam_service.server_multi_gpu:app --host "${WAM_HOST}" --port "${WAM_PORT}"
) >>"${WAM_LOG}" 2>&1 &
WAM_SERVER_PID=$!
echo "[run] WAM server pid=${WAM_SERVER_PID} log=${WAM_LOG} port=${WAM_PORT:-8006}"

cleanup() {
  if [[ -n "${WAM_SERVER_PID:-}" ]] && kill -0 "${WAM_SERVER_PID}" 2>/dev/null; then
    echo "[cleanup] stopping WAM service pid=${WAM_SERVER_PID}"
    kill "${WAM_SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 "${WAM_SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export WAM_HEALTH_URL="http://127.0.0.1:${WAM_PORT:-8006}/health"
bash "${COMMON_ROOT}/scripts/wait_wam_ready.sh"

bash "${PROJECT_ROOT}/RoboTwin_client.sh"
