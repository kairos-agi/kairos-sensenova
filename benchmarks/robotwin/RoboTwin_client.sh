#!/usr/bin/env bash
# Run RoboTwin eval against the kairos WAM service.
# Usage:
#   bash start_client.sh
# Key env vars:
#   MODEL_ENDPOINT           - WAM service URL (default: http://127.0.0.1:${WAM_PORT})
#   DATASET_STATS_PATH       - path to dataset_stats.json
#   EVAL_OUTPUT_DIR          - output directory
#   EVAL_GPU_IDS             - GPU IDs for eval workers (default: 0,1,2,3)
#   EVAL_NUM_EPISODES        - episodes per task per phase (default: 10)
#   CHUNK_SIZE               - episodes per chunk (default: 5)
#   RESUME                   - resume from existing output (default: false)

set -euo pipefail

_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${_project_dir}}"
export KAIROS_WAM_ROOT="${KAIROS_WAM_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${PROJECT_ROOT}/third_party/RoboTwin}"
export EVAL_STEP_LIMIT_FILE="${EVAL_STEP_LIMIT_FILE:-${PROJECT_ROOT}/robotwin_eval/config/_eval_step_limit.yml}"
export WAM_PORT="${WAM_PORT:-8006}"
export PYTHONPATH="${KAIROS_WAM_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:${WAM_PORT:-8006}}"
export DATASET_STATS_PATH="${DATASET_STATS_PATH:-}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
export DATASET_STATS_PATH="${DATASET_STATS_PATH:?DATASET_STATS_PATH must be set}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:?EVAL_OUTPUT_DIR must be set}"

export CLIENT_PYTHON="${CLIENT_PYTHON:?CLIENT_PYTHON must be set}"

if ! "${CLIENT_PYTHON}" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('sapien') and importlib.util.find_spec('yaml') else 1)" 2>/dev/null; then
  echo "[start_client] ERROR: CLIENT_PYTHON missing sapien or yaml: ${CLIENT_PYTHON}" >&2
  exit 1
fi

export EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-5}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-100}"
export CHUNK_SIZE="${CHUNK_SIZE:-5}"
export CHUNK_SEED_CANDIDATE_MULTIPLIER="${CHUNK_SEED_CANDIDATE_MULTIPLIER:-10}"
export RESUME="${RESUME:-false}"

export ACTION_HORIZON="${ACTION_HORIZON:-32}"
export REPLAN_STEPS="${REPLAN_STEPS:-24}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
export CFG_SCALE="${CFG_SCALE:-8.0}"

RESUME_ARG="--no-resume"
if [ "${RESUME}" = "true" ]; then
  RESUME_ARG="--resume"
fi

cd "${PROJECT_ROOT}"
exec "${CLIENT_PYTHON}" -m robotwin_eval.run_chunked_eval \
  --robotwin-root "${ROBOTWIN_ROOT}" \
  --eval-step-limit-file "${EVAL_STEP_LIMIT_FILE}" \
  --endpoint "${MODEL_ENDPOINT}" \
  --dataset-stats-path "${DATASET_STATS_PATH}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --eval-num-episodes "${EVAL_NUM_EPISODES}" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-seed-candidate-multiplier "${CHUNK_SEED_CANDIDATE_MULTIPLIER}" \
  "${RESUME_ARG}" \
  --gpu-ids "${EVAL_GPU_IDS}" \
  --max-tasks-per-gpu "${MAX_TASKS_PER_GPU}" \
  --action-horizon "${ACTION_HORIZON}" \
  --replan-steps "${REPLAN_STEPS}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --cfg-scale "${CFG_SCALE}"
