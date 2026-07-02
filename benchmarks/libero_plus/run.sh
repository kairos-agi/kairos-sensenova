#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMMON_ROOT="$(cd "${PROJECT_ROOT}/../common" && pwd)"
export KAIROS_WAM_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
export KAIROS_WAM_BENCH_ROOT="${PROJECT_ROOT}/kairos_wam"

# Shared paths and interpreters.
export KAIROS_MODEL_DIR="${KAIROS_MODEL_DIR:-/path/to/kairos_model}"
export LIBERO_PKG_ROOT="${LIBERO_PKG_ROOT:-${PROJECT_ROOT}/third_party/LIBERO-plus}"
export SERVER_PYTHON="${SERVER_PYTHON:-/path/to/kairos-wam/bin/python}"
export KAIROS_WAM_PYTHON="${KAIROS_WAM_PYTHON:-${SERVER_PYTHON}}"
export EVAL_PYTHON="${EVAL_PYTHON:-/path/to/libero-plus-eval/bin/python}"
export PYTHONPATH="${KAIROS_WAM_ROOT}:${KAIROS_WAM_BENCH_ROOT}:${KAIROS_WAM_BENCH_ROOT}/src:${LIBERO_PKG_ROOT}:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${KAIROS_WAM_BENCH_ROOT}/checkpoints}"
ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
ORIGINAL_NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-}"

# Work directories and LIBERO config.
_default_output_root="${PROJECT_ROOT}/../../outputs/libero_plus"
mkdir -p "${_default_output_root}"
_default_output_root="$(cd "${_default_output_root}" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${_default_output_root}}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${OUTPUT_ROOT}/eval}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${OUTPUT_ROOT}/libero_config}"
mkdir -p "${OUTPUT_ROOT}" "${EVAL_OUTPUT_ROOT}" "${LIBERO_CONFIG_PATH}" "${OUTPUT_ROOT}/libero_datasets"
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  cat >"${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${LIBERO_PKG_ROOT}/libero/libero
bddl_files: ${LIBERO_PKG_ROOT}/libero/libero/bddl_files
init_states: ${LIBERO_PKG_ROOT}/libero/libero/init_files
datasets: ${OUTPUT_ROOT}/libero_datasets
assets: ${LIBERO_PKG_ROOT}/libero/libero/assets
EOF
fi

# MuJoCo / EGL is fragile when multiple LIBERO workers initialize rendering at
# the same time. Keep a shared lock and retry policy available to all eval
# workers; libero_utils.py uses these values around OffScreenRenderEnv creation.
export LIBERO_EGL_INIT_LOCK="${LIBERO_EGL_INIT_LOCK:-1}"
export LIBERO_EGL_INIT_LOCK_FILE="${LIBERO_EGL_INIT_LOCK_FILE:-${OUTPUT_ROOT}/mujoco_egl_init.lock}"
export LIBERO_EGL_INIT_RETRIES="${LIBERO_EGL_INIT_RETRIES:-5}"
export LIBERO_EGL_INIT_RETRY_SLEEP_SEC="${LIBERO_EGL_INIT_RETRY_SLEEP_SEC:-5}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Category selection. CLI category/run-id arguments override this list:
#   bash run.sh "Background Textures" "Sensor Noise"
#   bash run.sh run1 run7
# Leave empty to run every configured category in run1..run7 order.
CATEGORIES=(
  "Background Textures"
  "Camera Viewpoints"
  "Objects Layout"
  "Robot Initial States"
  "Language Instructions"
  "Light Conditions"
  "Sensor Noise"
)

# Per-category defaults copied from run1.sh..run7.sh.
# Fields:
#   run_job|category|WAM_GPU_IDS|EVAL_GPU_IDS|MULTIRUN_NUM_GPUS|MAX_TASKS_PER_GPU|WAM_PRETRAINED_DIT
DEFAULT_DIT="/path/to/step-XXXXX-ema.safetensors"
CATEGORY_SPECS=(
  "run1|Background Textures|0,0,1,1|5,6|2|1|${DEFAULT_DIT}"
  "run2|Camera Viewpoints|0,0,1,1|0,1|2|1|${DEFAULT_DIT}"
  "run3|Objects Layout|0,0,1,1|0,1|2|1|${DEFAULT_DIT}"
  "run4|Robot Initial States|0,0,1,1,2,2,3,3|0,1,2,3|4|1|${DEFAULT_DIT}"
  "run5|Language Instructions|0,0,1,1|0,1|2|1|${DEFAULT_DIT}"
  "run6|Light Conditions|0,0,1,1|0,1|2|1|${DEFAULT_DIT}"
  "run7|Sensor Noise|0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7|0,1,2,3,4,5,6,7|8|1|${DEFAULT_DIT}"
)

# WAM server options. Per-category WAM_GPU_IDS and WAM_PRETRAINED_DIT are filled
# from CATEGORY_SPECS inside run_category().
export WAM_HOST="${WAM_HOST:-0.0.0.0}"
export WAM_PORT="${WAM_PORT:-8005}"
export WAM_CFG_PATH="${WAM_CFG_PATH:-${PROJECT_ROOT}/configs/libero_wam_infer_config.py}"
export WAM_EAGER_LOAD_ON_STARTUP="${WAM_EAGER_LOAD_ON_STARTUP:-1}"
export WAM_SKIP_LOAD_ENGINE="${WAM_SKIP_LOAD_ENGINE:-1}"
export WAM_WORKER_STARTUP_TIMEOUT_SEC="${WAM_WORKER_STARTUP_TIMEOUT_SEC:-600}"
export WAM_READY_TIMEOUT_SEC="${WAM_READY_TIMEOUT_SEC:-1200}"
export WAM_READY_POLL_SEC="${WAM_READY_POLL_SEC:-5}"
export QW_DROP_IMG_IN_VLM="${QW_DROP_IMG_IN_VLM:-1}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS:-ATEN,TRITON,CPP}"

# Evaluation/client options. Per-category EVAL_GPU_IDS, MULTIRUN_NUM_GPUS, and
# MAX_TASKS_PER_GPU are filled from CATEGORY_SPECS inside run_category().
export TASK_CONFIG="${TASK_CONFIG:-libero_uncond_2cam224_1e-4}"
export MODEL_CONFIG="${MODEL_CONFIG:-external_model_adapter}"
export CKPT="${CKPT:-/nothing/to/load}"
export DATASET_STATS_PATH="${DATASET_STATS_PATH:-${PROJECT_ROOT}/libero_plus_dataset_stats.json}"
export NUM_TRIALS="${NUM_TRIALS:-50}"
export EVAL_SUITE="${EVAL_SUITE:-libero_mix}"
export START_FROM="${START_FROM:-}"
export CREATE_LISTS_ONLY="${CREATE_LISTS_ONLY:-0}"
export SKIP_SUMMARIZE="${SKIP_SUMMARIZE:-0}"
export STRICT_SEQUENTIAL="${STRICT_SEQUENTIAL:-0}"
export LAUNCH_STALE_SEC="${LAUNCH_STALE_SEC:-180}"
export LAUNCH_VIA_TMUX="${LAUNCH_VIA_TMUX:-0}"
export GHOST_RECLAIM_SEC="${GHOST_RECLAIM_SEC:-0}"
export MONITORING_INTERVAL="${MONITORING_INTERVAL:-1}"
export STATUS_INTERVAL="${STATUS_INTERVAL:-10}"

usage() {
  cat <<'EOF'
Usage:
  bash run.sh [options] [CATEGORY_OR_RUN_ID ...]

Options:
  --list                    Print category/GPU defaults and exit.
  -h, --help                Show this help.

Examples:
  bash run.sh
  bash run.sh "Background Textures" "Sensor Noise"
  bash run.sh run1 run7
EOF
}

list_categories() {
  local spec run_job category wam_gpu_ids eval_gpus num_gpus max_tasks_per_gpu dit
  for spec in "${CATEGORY_SPECS[@]}"; do
    IFS='|' read -r run_job category wam_gpu_ids eval_gpus num_gpus max_tasks_per_gpu dit <<<"${spec}"
    printf '%-5s | %-22s | WAM_GPU_IDS=%-35s | EVAL_GPU_IDS=%-17s | NUM_GPUS=%s | MAX_TASKS_PER_GPU=%s\n' \
      "${run_job}" "${category}" "${wam_gpu_ids}" "${eval_gpus}" "${num_gpus}" "${max_tasks_per_gpu}"
  done
}

detect_visible_gpu_count() {
  local count

  if [[ -n "${EVAL_VISIBLE_GPU_COUNT_OVERRIDE:-}" ]]; then
    printf '%s\n' "${EVAL_VISIBLE_GPU_COUNT_OVERRIDE}"
    return 0
  fi

  count="$(cuda_device_count_current)"
  if [[ "${count}" =~ ^[0-9]+$ ]] && [[ "${count}" -gt 0 ]]; then
    printf '%s\n' "${count}"
    return 0
  fi

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    count="$(count_gpu_id_entries "${CUDA_VISIBLE_DEVICES}")"
    if [[ "${count}" =~ ^[0-9]+$ ]] && [[ "${count}" -gt 0 ]]; then
      printf '%s\n' "${count}"
      return 0
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${count}" =~ ^[0-9]+$ ]] && [[ "${count}" -gt 0 ]]; then
      printf '%s\n' "${count}"
      return 0
    fi
  fi

  printf '0\n'
}

cuda_device_count_current() {
  local count
  count="$("${EVAL_PYTHON}" - <<'PY' 2>/dev/null || true
import torch
print(torch.cuda.device_count())
PY
)"
  count="$(echo "${count}" | tail -n 1 | tr -d '[:space:]')"
  if [[ "${count}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${count}"
  else
    printf '0\n'
  fi
}

cuda_device_count_for_gpu_ids() {
  local gpu_ids="$1"
  local count
  count="$(CUDA_VISIBLE_DEVICES="${gpu_ids}" "${EVAL_PYTHON}" - <<'PY' 2>/dev/null || true
import torch
print(torch.cuda.device_count())
PY
)"
  count="$(echo "${count}" | tail -n 1 | tr -d '[:space:]')"
  if [[ "${count}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${count}"
  else
    printf '0\n'
  fi
}

count_gpu_id_entries() {
  local gpu_ids="$1"
  local count=0
  local gpu

  IFS=',' read -r -a _gpu_id_parts <<<"${gpu_ids}"
  for gpu in "${_gpu_id_parts[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ -n "${gpu}" && "${gpu}" != "-1" && "${gpu}" != "NoDevFiles" && "${gpu}" != "none" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

gpu_ids_valid_for_visible_count() {
  local gpu_ids="$1"
  local visible_count="$2"
  local gpu

  if [[ ! "${visible_count}" =~ ^[0-9]+$ ]] || [[ "${visible_count}" -le 0 ]]; then
    return 1
  fi

  IFS=',' read -r -a _gpu_id_parts <<<"${gpu_ids}"
  for gpu in "${_gpu_id_parts[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ -z "${gpu}" ]]; then
      return 1
    fi
    # UUID / MIG identifiers cannot be range-checked here; let CUDA handle them.
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      return 0
    fi
    if [[ "${gpu}" -ge "${visible_count}" ]]; then
      return 1
    fi
  done
  return 0
}

logical_gpu_ids() {
  local count="$1"
  seq 0 "$((count - 1))" | paste -sd, -
}

resolve_eval_gpu_ids() {
  local requested="$1"
  local num_gpus="$2"
  local visible_count
  local requested_count

  if [[ -n "${EVAL_GPU_IDS_OVERRIDE:-}" ]]; then
    requested="${EVAL_GPU_IDS_OVERRIDE}"
  fi
  if [[ "${EVAL_GPU_ID_MODE:-auto}" == "as_is" ]]; then
    printf '%s\n' "${requested}"
    return 0
  fi

  requested_count="$(cuda_device_count_for_gpu_ids "${requested}")"
  if [[ "${requested_count}" =~ ^[0-9]+$ ]] && [[ "${requested_count}" -ge "${num_gpus}" ]]; then
    printf '%s\n' "${requested}"
    return 0
  fi

  visible_count="$(detect_visible_gpu_count)"
  if [[ ! "${visible_count}" =~ ^[0-9]+$ ]]; then
    echo "[gpu] ERROR: invalid visible GPU count detected: ${visible_count}" >&2
    exit 2
  fi

  if [[ "${visible_count}" -ge "${num_gpus}" ]]; then
    local remapped
    local remapped_count
    remapped="$(logical_gpu_ids "${num_gpus}")"
    remapped_count="$(cuda_device_count_for_gpu_ids "${remapped}")"
    if [[ ! "${remapped_count}" =~ ^[0-9]+$ ]] || [[ "${remapped_count}" -lt "${num_gpus}" ]]; then
      echo "[gpu] ERROR: remapped EVAL_GPU_IDS=${remapped} exposes only ${remapped_count}/${num_gpus} CUDA device(s)" >&2
      exit 2
    fi
    echo "[gpu] requested EVAL_GPU_IDS=${requested} exposes ${requested_count}/${num_gpus} CUDA device(s); current container exposes ${visible_count}. Remap eval GPUs to ${remapped}" >&2
    printf '%s\n' "${remapped}"
    return 0
  fi

  if gpu_ids_valid_for_visible_count "${requested}" "${visible_count}"; then
    printf '%s\n' "${requested}"
    return 0
  fi

  echo "[gpu] ERROR: requested EVAL_GPU_IDS=${requested} exposes ${requested_count}/${num_gpus} CUDA device(s), and current container exposes only ${visible_count}" >&2
  echo "[gpu] Set EVAL_GPU_IDS_OVERRIDE=... or EVAL_GPU_ID_MODE=as_is if this detection is wrong." >&2
  exit 2
}

find_category_spec() {
  local wanted="$1"
  local spec run_job category wam_gpu_ids eval_gpus num_gpus max_tasks_per_gpu dit

  for spec in "${CATEGORY_SPECS[@]}"; do
    IFS='|' read -r run_job category wam_gpu_ids eval_gpus num_gpus max_tasks_per_gpu dit <<<"${spec}"
    if [[ "${wanted}" == "${category}" || "${wanted}" == "${run_job}" || "${wanted}" == "${run_job#run}" ]]; then
      printf '%s\n' "${spec}"
      return 0
    fi
  done

  return 1
}

append_all_specs() {
  local spec
  for spec in "${CATEGORY_SPECS[@]}"; do
    SELECTED_SPECS+=("${spec}")
  done
}

validate_checkpoint() {
  if [[ -z "${WAM_PRETRAINED_DIT:-}" ]]; then
    echo "[${RUN_JOB}] ERROR: WAM_PRETRAINED_DIT is not set" >&2
    exit 1
  fi
  if [[ ! -f "${WAM_PRETRAINED_DIT}" ]]; then
    echo "[${RUN_JOB}] ERROR: checkpoint not found: ${WAM_PRETRAINED_DIT}" >&2
    exit 1
  fi
}

cleanup_server() {
  if [[ -n "${WAM_SERVER_PID:-}" ]] && kill -0 "${WAM_SERVER_PID}" 2>/dev/null; then
    echo "[cleanup] stopping WAM service pid=${WAM_SERVER_PID}"
    kill "${WAM_SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 "${WAM_SERVER_PID}" 2>/dev/null || true
  fi
}

source "${COMMON_ROOT}/scripts/egl_preflight.sh"

CLI_CATEGORIES=()
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        CLI_CATEGORIES+=("$1")
        shift
      done
      ;;
    *)
      CLI_CATEGORIES+=("$1")
      shift
      ;;
  esac
done

if [[ "${LIST_ONLY}" == "1" ]]; then
  list_categories
  exit 0
fi

REQUESTED_CATEGORIES=()
if [[ ${#CLI_CATEGORIES[@]} -gt 0 ]]; then
  REQUESTED_CATEGORIES=("${CLI_CATEGORIES[@]}")
else
  REQUESTED_CATEGORIES=("${CATEGORIES[@]}")
fi

SELECTED_SPECS=()
if [[ ${#REQUESTED_CATEGORIES[@]} -eq 0 ]]; then
  append_all_specs
else
  for category in "${REQUESTED_CATEGORIES[@]}"; do
    if [[ "${category}" == "all" ]]; then
      append_all_specs
      continue
    fi

    if ! spec="$(find_category_spec "${category}")"; then
      echo "[run_all] ERROR: unknown category or run id: ${category}" >&2
      echo "[run_all] Available categories:" >&2
      list_categories >&2
      exit 2
    fi
    SELECTED_SPECS+=("${spec}")
  done
fi

run_category() (
  local spec="$1"
  local run_job category wam_gpu_ids eval_gpus num_gpus max_tasks_per_gpu default_dit
  local run_ts workdir batch_output_dir wam_log
  local -a server_args client_args

  IFS='|' read -r run_job category wam_gpu_ids eval_gpus num_gpus max_tasks_per_gpu default_dit <<<"${spec}"

  export RUN_JOB="${run_job}"
  export WAM_GPU_IDS="${wam_gpu_ids}"
  export WAM_PRETRAINED_DIT="${WAM_PRETRAINED_DIT:-${default_dit}}"
  validate_checkpoint
  local requested_eval_gpus="${eval_gpus}"
  eval_gpus="$(resolve_eval_gpu_ids "${requested_eval_gpus}" "${num_gpus}")"

  run_ts="${RUN_JOB}_$(date +%Y%m%d_%H%M%S)"
  workdir="${OUTPUT_ROOT}/${RUN_JOB}"
  wam_log="${WAM_LOG:-${workdir}/wam_server.log}"
  batch_output_dir="${EVAL_OUTPUT_DIR:-${EVAL_OUTPUT_ROOT}/${TASK_CONFIG}/by_category_${run_ts}}"

  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${workdir}/torch_inductor_cache}"
  export BATCH_TS="${BATCH_TS:-${run_ts}}"
  export CUDA_VISIBLE_DEVICES="${eval_gpus}"
  export EVAL_GPU_IDS="${eval_gpus}"
  # The underlying LIBERO scheduler still reads this internal variable.
  export EVAL_GPUS="${EVAL_GPU_IDS}"
  export NUM_GPUS="${num_gpus}"
  export MULTIRUN_NUM_GPUS="${num_gpus}"
  export MAX_TASKS_PER_GPU="${max_tasks_per_gpu}"
  export MAX_LAUNCH_PER_ROUND="${MAX_LAUNCH_PER_ROUND:-$((NUM_GPUS * MAX_TASKS_PER_GPU))}"
  export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:${WAM_PORT}}"
  export WAM_HEALTH_URL="${WAM_HEALTH_URL:-http://127.0.0.1:${WAM_PORT}/health}"
  export EVAL_OUTPUT_DIR="${batch_output_dir}"

  mkdir -p "${workdir}" "${TORCHINDUCTOR_CACHE_DIR}" "$(dirname "${wam_log}")" "${batch_output_dir}"

  echo "[${RUN_JOB}] category=${category}"
  echo "[${RUN_JOB}] server_python=${SERVER_PYTHON}"
  echo "[${RUN_JOB}] eval_python=${EVAL_PYTHON}"
  echo "[${RUN_JOB}] wam_gpus=${WAM_GPU_IDS} requested_eval_gpu_ids=${requested_eval_gpus} eval_gpu_ids=${EVAL_GPU_IDS} endpoint=${MODEL_ENDPOINT}"
  echo "[${RUN_JOB}] inherited CUDA_VISIBLE_DEVICES=${ORIGINAL_CUDA_VISIBLE_DEVICES} NVIDIA_VISIBLE_DEVICES=${ORIGINAL_NVIDIA_VISIBLE_DEVICES}"
  echo "[${RUN_JOB}] checkpoint=${WAM_PRETRAINED_DIT}"
  echo "[${RUN_JOB}] output_root=${OUTPUT_ROOT}"
  echo "[${RUN_JOB}] eval_output_dir=${batch_output_dir}"
  echo "[${RUN_JOB}] dataset_stats_path=${DATASET_STATS_PATH}"
  echo "[${RUN_JOB}] mujoco_gl=${MUJOCO_GL} pyopengl_platform=${PYOPENGL_PLATFORM} egl_vendor=${LIBERO_EGL_VENDOR_MODE_RESOLVED:-unset} egl_device_id_mode=${LIBERO_EGL_DEVICE_ID_MODE:-local}"

  libero_preflight_egl_devices
  libero_preflight_worker_render_contexts "${RUN_JOB}" "${EVAL_GPU_IDS}"

  server_args=(--host "${WAM_HOST}" --port "${WAM_PORT}")
  (
    cd "${COMMON_ROOT}"
    exec "${SERVER_PYTHON}" -m uvicorn wam_service.server_multi_gpu:app "${server_args[@]}"
  ) >>"${wam_log}" 2>&1 &
  WAM_SERVER_PID=$!
  echo "[${RUN_JOB}] WAM server pid=${WAM_SERVER_PID} log=${wam_log}"

  trap cleanup_server EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  bash "${COMMON_ROOT}/scripts/wait_wam_ready.sh"

  client_args=(
    "${KAIROS_WAM_BENCH_ROOT}/experiments/libero/run_libero_plus_by_category.py"
    --output-dir "${batch_output_dir}"
    --suite "${EVAL_SUITE}"
    --categories "${category}"
  )
  if [[ -n "${START_FROM}" ]]; then
    client_args+=(--start-from "${START_FROM}")
  fi
  if [[ "${CREATE_LISTS_ONLY}" == "1" ]]; then
    client_args+=(--create-lists-only)
  fi
  if [[ "${SKIP_SUMMARIZE}" == "1" ]]; then
    client_args+=(--skip-summarize)
  fi
  client_args+=(
    --
    "task=${TASK_CONFIG}"
    "model=${MODEL_CONFIG}"
    "model.endpoint=${MODEL_ENDPOINT}"
    "ckpt=${CKPT}"
    "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}"
    "EVALUATION.num_trials=${NUM_TRIALS}"
    "MULTIRUN.num_gpus=${MULTIRUN_NUM_GPUS}"
    "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
  )

  cd "${KAIROS_WAM_BENCH_ROOT}"
  "${EVAL_PYTHON}" "${client_args[@]}"
)

echo "[run_all] selected categories:"
for spec in "${SELECTED_SPECS[@]}"; do
  IFS='|' read -r run_job category wam_gpu_ids eval_gpus num_gpus max_tasks_per_gpu dit <<<"${spec}"
  echo "  - ${run_job}: ${category} (WAM_GPU_IDS=${wam_gpu_ids}, EVAL_GPU_IDS=${eval_gpus}, NUM_GPUS=${num_gpus}, MAX_TASKS_PER_GPU=${max_tasks_per_gpu})"
done

libero_configure_egl_backend

for spec in "${SELECTED_SPECS[@]}"; do
  run_category "${spec}"
done
