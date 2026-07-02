#!/bin/bash

# This script runs LIBERO evaluation tasks in parallel.
# v3: dynamic GPU load management.

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR="${ROOT_DIR:-$(cd "${_script_dir}/../.." && pwd)}"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${ROOT_DIR}/.." && pwd)}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT_DIR}/checkpoints}"
export KAIROS_WAM_ROOT="${KAIROS_WAM_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}"
export LIBERO_PKG_ROOT="${LIBERO_PKG_ROOT:?LIBERO_PKG_ROOT must be set by libero_plus/run.sh}"
export EVAL_PYTHON="${EVAL_PYTHON:?EVAL_PYTHON must be set by libero_plus/run.sh}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_EGL_DEVICE_ID_MODE="${LIBERO_EGL_DEVICE_ID_MODE:-local}"

_eval_python_ready() {
    "${EVAL_PYTHON}" -c "
import importlib.util
mods = ('hydra', 'robosuite')
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
" 2>/dev/null
}

if ! _eval_python_ready; then
    echo "ERROR: EVAL_PYTHON does not provide hydra + robosuite: ${EVAL_PYTHON}" >&2
    exit 1
fi


run_libero_eval() {
    local task_list_file=$1
    echo "task_file: $task_list_file"

    require_non_empty() {
        local var_name="$1"
        local var_val="${!var_name}"
        if [ -z "$var_val" ]; then
            echo "Error: required variable $var_name is not set"
            exit 1
        fi
    }
    
    # Basic configuration
    ROOT_DIR=${ROOT_DIR:-"$(pwd)"}
    export ROOT_DIR
    # Fixed PYTHONPATH for workers (do not append $PYTHONPATH — avoids tmux env bloat).
    EVAL_PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src:${LIBERO_PKG_ROOT}"
    export PYTHONPATH="${EVAL_PYTHONPATH}"
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${ROOT_DIR}/configs}"
    export LIBERO_CONFIG_PATH
    EVAL_PYTHON="${EVAL_PYTHON:-$(command -v python)}"
    export EVAL_PYTHON
    # Generate a unique run_id
    RUN_ID=${RUN_ID:-"eval_$(date +%Y%m%d_%H%M%S)"}
    export RUN_ID
    OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/evaluate_results/$RUN_ID"}
    export OUTPUT_DIR  # Use run_id as the output subdirectory
    SESSION_NAME="libero_test_v3"
    EXP_NAME=${EXP_NAME:-""}
    export EXP_NAME
    PANE_INIT_DIR="$OUTPUT_DIR/.pane_env_init"

    echo "EXP_NAME: $EXP_NAME"
    
    # Create the output directory
    mkdir -p "$OUTPUT_DIR" "$PANE_INIT_DIR"
    echo "Evaluation results will be saved to: $OUTPUT_DIR"

    # One scheduler per OUTPUT_DIR (avoid duplicate schedulers racing on state files).
    local lock_file="$OUTPUT_DIR/.scheduler.lock"
    exec 200>"$lock_file"
    if ! flock -n 200; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Another scheduler is already running for OUTPUT_DIR=$OUTPUT_DIR"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Lock: $lock_file"
        exit 2
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scheduler lock acquired (pid=$$): $lock_file"

    # Copy task_list_file into OUTPUT_DIR
    cp "$task_list_file" "$OUTPUT_DIR/"
    task_list_file="$OUTPUT_DIR/$(basename $task_list_file)"
    echo "Task list file copied to: $task_list_file"

    # Match eval_libero_single.py: libero_mix results live under <suite>/<category_dir>/
    sanitize_category_dir() {
        local raw="$1"
        raw="${raw#"${raw%%[![:space:]]*}"}"
        raw="${raw%"${raw##*[![:space:]]}"}"
        echo "${raw// /_}"
    }

    resolve_libero_category_value() {
        if [ -n "${LIBERO_CATEGORY_VALUE:-}" ]; then
            echo "$LIBERO_CATEGORY_VALUE"
            return 0
        fi
        if [ -z "${EXTRA_ARGS:-}" ]; then
            return 1
        fi
        local token
        for token in $EXTRA_ARGS; do
            case "$token" in
                EVALUATION.category_value=*)
                    echo "${token#EVALUATION.category_value=}"
                    return 0
                    ;;
            esac
        done
        return 1
    }

    get_suite_result_dir() {
        local suite="$1"
        local category_value="${2:-}"
        if [ -z "$category_value" ]; then
            category_value=$(resolve_libero_category_value 2>/dev/null || true)
        fi
        if [ -n "$category_value" ]; then
            local category_dir
            category_dir=$(sanitize_category_dir "$category_value")
            echo "$OUTPUT_DIR/$suite/$category_dir"
            return 0
        fi
        echo "$OUTPUT_DIR/$suite"
    }

    make_task_key() {
        local suite="$1"
        local task_id="$2"
        local category="${3:-}"
        if [ -n "$category" ]; then
            echo "${suite},${task_id},${category}"
        else
            echo "${suite},${task_id}"
        fi
    }

    parse_task_fields() {
        local line="$1"
        PARSED_SUITE=$(echo "$line" | cut -d, -f1)
        PARSED_TASK_ID=$(echo "$line" | cut -d, -f2)
        PARSED_CATEGORY=$(echo "$line" | cut -d, -f3-)
        if [ -z "$PARSED_CATEGORY" ]; then
            PARSED_CATEGORY=""
        fi
    }

    task_status_tag() {
        local suite="$1"
        local task_id="$2"
        local category="${3:-}"
        if [ -n "$category" ]; then
            local category_dir
            category_dir=$(sanitize_category_dir "$category")
            echo "${suite}_${category_dir}_task${task_id}"
        else
            echo "${suite}_task${task_id}"
        fi
    }
    
    # GPU and tmux configuration.
    # Scheduler uses logical slot IDs (0..N-1), while workers launch with the
    # corresponding physical CUDA_VISIBLE_DEVICES entry.
    PHYSICAL_GPU_ARRAY=()
    PHYSICAL_GPU_LABEL_ARRAY=()
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # If CUDA_VISIBLE_DEVICES is not set, require NUM_GPUS explicitly
        require_non_empty "NUM_GPUS"
        AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
    else
        IFS=',' read -r -a PHYSICAL_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
        NUM_GPUS=${#PHYSICAL_GPU_ARRAY[@]}
        AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
        PHYSICAL_GPU_LABEL_ARRAY=("${PHYSICAL_GPU_ARRAY[@]}")
    fi
    export NUM_GPUS
    echo "NUM_GPUS: $NUM_GPUS, scheduler slots: $AVAILABLE_GPUS, physical GPUs: ${PHYSICAL_GPU_LABEL_ARRAY[*]:-$AVAILABLE_GPUS}"

    # Convert AVAILABLE_GPUS to an array (logical slot indices for load balancing)
    IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"

    resolve_physical_gpu() {
        local logical_id=$1
        if [ ${#PHYSICAL_GPU_LABEL_ARRAY[@]} -gt 0 ]; then
            echo "${PHYSICAL_GPU_LABEL_ARRAY[$logical_id]}"
        else
            echo "$logical_id"
        fi
    }

    gpu_slot_label() {
        local logical_id=$1
        local physical
        physical=$(resolve_physical_gpu "$logical_id")
        if [ "$physical" != "$logical_id" ]; then
            echo "${physical}(slot${logical_id})"
        else
            echo "$logical_id"
        fi
    }

    require_non_empty "MAX_TASKS_PER_GPU"
    require_non_empty "NUM_TRIALS"
    TMUX_GRID_ROWS=${TMUX_GRID_ROWS:-1}
    TMUX_GRID_COLS=${TMUX_GRID_COLS:-$((MAX_TASKS_PER_GPU + 1))}
    GRID_ROWS=$TMUX_GRID_ROWS
    GRID_COLS=$TMUX_GRID_COLS
    MAX_PANES=$((GRID_ROWS * GRID_COLS - 1))
    if [ "$MAX_PANES" -le 0 ]; then
        echo "Error: invalid tmux grid configuration, TMUX_GRID_ROWS=$TMUX_GRID_ROWS TMUX_GRID_COLS=$TMUX_GRID_COLS"
        exit 1
    fi
    
    # GPU load tracking files
    GPU_LOAD_FILE="$OUTPUT_DIR/gpu_load.txt"
    TASK_GPU_MAP_FILE="$OUTPUT_DIR/task_gpu_map.txt"
    TASK_STATUS_DIR="$OUTPUT_DIR/task_status"
    TASK_LOG_DIR="$OUTPUT_DIR/task_logs"
    FAILED_TASKS_FILE="$OUTPUT_DIR/failed_tasks.txt"

    mkdir -p "$TASK_STATUS_DIR" "$TASK_LOG_DIR"
    : > "$FAILED_TASKS_FILE"
    
    # Initialize GPU load tracking
    init_gpu_load_tracking() {
        # Initialize the current task count of each GPU to 0
        > "$GPU_LOAD_FILE"
        > "$TASK_GPU_MAP_FILE"
        for gpu in "${GPU_ARRAY[@]}"; do
            echo "$gpu:0" >> "$GPU_LOAD_FILE"
        done
        echo "GPU load tracking initialized: $GPU_LOAD_FILE"
    }
    
    # Get the current GPU load
    get_gpu_load() {
        local gpu_id=$1
        local load=$(grep "^$gpu_id:" "$GPU_LOAD_FILE" | cut -d: -f2)
        echo "${load:-0}"
    }
    
    # Update GPU load
    update_gpu_load() {
        local gpu_id=$1
        local new_load=$2
        # Use a temporary file to keep the update atomic
        local temp_file="$GPU_LOAD_FILE.tmp"
        
        # Check whether the file exists first
        if [ -f "$GPU_LOAD_FILE" ]; then
            # Remove the old record and keep records for other GPUs
            grep -v "^${gpu_id}:" "$GPU_LOAD_FILE" > "$temp_file" 2>/dev/null || true
        else
            > "$temp_file"
        fi
        
        # Add the new record
        echo "${gpu_id}:${new_load}" >> "$temp_file"
        mv "$temp_file" "$GPU_LOAD_FILE"
    }
    
    # Increment GPU load
    increment_gpu_load() {
        local gpu_id=$1
        local current_load=$(get_gpu_load $gpu_id)
        local new_load=$((current_load + 1))
        update_gpu_load $gpu_id $new_load
        echo $new_load
    }
    
    # Decrement GPU load
    decrement_gpu_load() {
        local gpu_id=$1
        local current_load=$(get_gpu_load $gpu_id)
        local new_load=$((current_load - 1))
        [ $new_load -lt 0 ] && new_load=0
        update_gpu_load $gpu_id $new_load
        echo $new_load
    }
    
    # Find the least-loaded GPU
    find_least_loaded_gpu() {
        local min_load=999999
        local best_gpu=""
        for gpu in "${GPU_ARRAY[@]}"; do
            local load=$(get_gpu_load $gpu)
            if [ $load -lt $min_load ] && [ $load -lt $MAX_TASKS_PER_GPU ]; then
                min_load=$load
                best_gpu=$gpu
            fi
        done
        echo $best_gpu
    }
    
    # Show GPU load status
    show_gpu_status() {
        echo "=== GPU Load Status ==="
        for gpu in "${GPU_ARRAY[@]}"; do
            local load=$(get_gpu_load $gpu)
            local physical
            physical=$(resolve_physical_gpu "$gpu")
            local percentage=$((load * 100 / MAX_TASKS_PER_GPU))
            if [ "$physical" != "$gpu" ]; then
                printf "GPU %s (slot %s): %d/%d tasks (%d%%)\n" "$physical" "$gpu" "$load" "$MAX_TASKS_PER_GPU" "$percentage"
            else
                printf "GPU %s: %d/%d tasks (%d%%)\n" "$gpu" "$load" "$MAX_TASKS_PER_GPU" "$percentage"
            fi
        done
        echo "=================="
    }
    
    # Debug helper: show the currently running tasks
    show_debug_info() {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Debug Info ==="
        
        # Show the GPU load file contents
        if [ -f "$GPU_LOAD_FILE" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU load file contents:"
            cat "$GPU_LOAD_FILE" | while IFS=: read gpu load; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]   GPU$gpu: $load"
            done
        fi
        
        # Show the task mapping file contents
        if [ -f "$TASK_GPU_MAP_FILE" ]; then
            local map_count=$(wc -l < "$TASK_GPU_MAP_FILE" 2>/dev/null || echo 0)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Number of running tasks: $map_count"
            if [ $map_count -gt 0 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running tasks:"
                cat "$TASK_GPU_MAP_FILE" | while IFS=: read task_info gpu_id; do
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   $task_info -> GPU$gpu_id"
                done
            fi
        fi
        
        if [ -f "$PENDING_TASKS_FILE" ]; then
            local pending_queue pending_cursor pending_total
            pending_queue=$(get_pending_queue_count)
            pending_cursor=$(read_pending_cursor)
            pending_total=$(get_pending_line_count)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pending queue: $pending_queue (cursor=$pending_cursor/$pending_total)"
        fi

        if [ -f "$FAILED_TASKS_FILE" ]; then
            local failed_count=$(wc -l < "$FAILED_TASKS_FILE" 2>/dev/null || echo 0)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Number of failed tasks: $failed_count"
        fi
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==================="
    }
    
    # Record the task-to-GPU mapping
    record_task_gpu_mapping() {
        local suite=$1
        local task_id=$2
        local gpu_id=$3
        local category="${4:-}"
        local task_key
        task_key=$(make_task_key "$suite" "$task_id" "$category")
        echo "${task_key}:${gpu_id}" >> "$TASK_GPU_MAP_FILE"
    }
    
    # Get the GPU assigned to a task
    get_task_gpu() {
        local suite=$1
        local task_id=$2
        local category="${3:-}"
        local task_key
        task_key=$(make_task_key "$suite" "$task_id" "$category")
        local mapping
        mapping=$(grep -F "${task_key}:" "$TASK_GPU_MAP_FILE" | head -n 1)
        echo "${mapping#*:}"
    }
    
    # Remove the task-to-GPU mapping
    remove_task_gpu_mapping() {
        local suite=$1
        local task_id=$2
        local category="${3:-}"
        local task_key
        task_key=$(make_task_key "$suite" "$task_id" "$category")
        local temp_file="$TASK_GPU_MAP_FILE.tmp"
        grep -vF "${task_key}:" "$TASK_GPU_MAP_FILE" > "$temp_file" 2>/dev/null || true
        mv "$temp_file" "$TASK_GPU_MAP_FILE"
    }

    mark_task_failed() {
        local suite=$1
        local task_id=$2
        local gpu_id=$3
        local return_code=$4
        local log_file=$5
        local category="${6:-}"
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$timestamp,$suite,$task_id,$category,gpu=$gpu_id,rc=$return_code,log=$log_file" >> "$FAILED_TASKS_FILE"
    }
    
    # Checkpoint and config
    CKPT=${CKPT:-""}
    export CKPT
    CONFIG=${CONFIG:-""}
    require_non_empty "CKPT"
    require_non_empty "CONFIG"
    # Normalize CONFIG to task/config_name.yaml
    CONFIG="${CONFIG#configs/}" # delete prefix configs/
    CONFIG="${CONFIG#task/}" # delete prefix task/
    CONFIG="${CONFIG%.yaml}" # delete suffix .yaml
    export CONFIG

    echo "CKPT: $CKPT"
    echo "CONFIG: $CONFIG"
    echo "ROOT_DIR: $ROOT_DIR"
    echo "NUM_GPUS: $NUM_GPUS"
    echo "MAX_TASKS_PER_GPU: $MAX_TASKS_PER_GPU"
    
    # Initialize GPU load tracking
    init_gpu_load_tracking

    # Direct subprocess launch avoids tmux quoting/truncation (default). Set LAUNCH_VIA_TMUX=1 for pane UI.
    LAUNCH_VIA_TMUX="${LAUNCH_VIA_TMUX:-0}"
    # Do not reclaim pid=0 slots unless explicitly enabled (tmux-only failure mode).
    GHOST_RECLAIM_SEC="${GHOST_RECLAIM_SEC:-0}"

    if [ "$LAUNCH_VIA_TMUX" != "1" ]; then
        if tmux has-session -t $SESSION_NAME 2>/dev/null; then
            tmux kill-session -t $SESSION_NAME 2>/dev/null || true
            echo "Removed stale tmux session '$SESSION_NAME' (direct launch mode; attach old panes is misleading)."
        fi
    fi

    # Check for an existing tmux session
    if [ "$LAUNCH_VIA_TMUX" = "1" ] && tmux has-session -t $SESSION_NAME 2>/dev/null; then
        # If the session exists, delete it
        tmux kill-session -t $SESSION_NAME
        echo "Session '$SESSION_NAME' has been deleted"
    fi
    rm -rf "$PANE_INIT_DIR"
    mkdir -p "$PANE_INIT_DIR"

    # Create a new detached session (only when using tmux launch)
    if [ "$LAUNCH_VIA_TMUX" = "1" ]; then
        tmux new-session -d -s $SESSION_NAME
    fi

    # Create the grid layout
    create_grid_layout() {
        local window=$1
        if [ $window -gt 0 ]; then
            # Check whether the window exists
            if ! tmux list-windows -t $SESSION_NAME | grep -q "^$window:"; then
                tmux new-window -t $SESSION_NAME:$window
            fi
        fi
        
        # Get the current number of panes in the window
        local pane_count=$(tmux list-panes -t $SESSION_NAME:$window | wc -l)
        
        # Only create new panes if the current count is below the target count
        for ((i=pane_count; i<GRID_ROWS*GRID_COLS-1; i++)); do
            tmux split-window -t $SESSION_NAME:$window
            tmux select-layout -t $SESSION_NAME:$window tiled
        done
    }
    
    # Create the first window layout
    if [ "$LAUNCH_VIA_TMUX" = "1" ]; then
        create_grid_layout 0
    fi
    
    # Global pane counter
    NEXT_PANE_INDEX=0
    
    # Helper to ensure a window and pane exist
    ensure_pane_exists() {
        local window_id=$1
        local pane_id=$2
        
        # Ensure the window exists
        if [ $window_id -gt 0 ]; then
            if ! tmux list-windows -t $SESSION_NAME | grep -q "^$window_id:" 2>/dev/null; then
                tmux new-window -t $SESSION_NAME:$window_id 2>/dev/null
                create_grid_layout $window_id
            fi
        fi
        
        # If this is the first pane of a non-zero window, ensure the grid is created
        if [ $pane_id -eq 0 ] && [ $window_id -gt 0 ]; then
            create_grid_layout $window_id
        fi
    }
    
    pane_init_marker() {
        local pane_info=$1
        echo "$PANE_INIT_DIR/$(echo "$pane_info" | tr ':' '_').done"
    }

    # EXTRA_ARGS comes from Hydra overrides (string). When we already pass some overrides
    # explicitly (like EVALUATION.output_dir / EVALUATION.category_value), keeping duplicates
    # can massively bloat the tmux command and even get it truncated, leading to empty
    # redirection targets (e.g. "$STATUS_FILE" becoming empty) and confusing bash errors.
    sanitize_extra_args() {
        local token
        SANITIZED_EXTRA_ARGS=()
        if [ -z "${EXTRA_ARGS:-}" ]; then
            return 0
        fi
        for token in $EXTRA_ARGS; do
            case "$token" in
                EVALUATION.output_dir=*|EVALUATION.category_value=*)
                    continue
                    ;;
            esac
            SANITIZED_EXTRA_ARGS+=("$token")
        done
    }
    sanitize_extra_args

    # Extract a few stable overrides from EXTRA_ARGS (these are space-safe: no spaces in values).
    # Avoid forwarding the whole EXTRA_ARGS into tmux, because quoting/length issues can corrupt
    # the command line (especially for category_value with spaces).
    resolve_simple_override() {
        local key="$1"
        local raw="${EXTRA_ARGS:-}"
        # match key=value up to next space
        echo "$raw" | sed -n "s/.*\\(${key}=[^ ]*\\).*/\\1/p" | head -n 1
    }
    MODEL_ADAPTER_OVERRIDE="$(resolve_simple_override 'model')"
    MODEL_ENDPOINT_OVERRIDE="$(resolve_simple_override 'model.endpoint')"
    DATASET_STATS_OVERRIDE="$(resolve_simple_override 'EVALUATION.dataset_stats_path')"
    if [ -z "$MODEL_ADAPTER_OVERRIDE" ]; then
        MODEL_ADAPTER_OVERRIDE="model=external_model_adapter"
    fi

    # Source eval env once per tmux pane (repeated source blows up LD_LIBRARY_PATH).
    init_pane_env_once() {
        local pane_info=$1
        local marker
        marker=$(pane_init_marker "$pane_info")
        if [ -f "$marker" ]; then
            return 0
        fi
        tmux send-keys -t "$SESSION_NAME:$pane_info" \
            "export PYTHONPATH=\"$EVAL_PYTHONPATH\" && export LIBERO_CONFIG_PATH=\"$LIBERO_CONFIG_PATH\" && export MUJOCO_GL=\"$MUJOCO_GL\" && export PYOPENGL_PLATFORM=\"$PYOPENGL_PLATFORM\" && export LIBERO_EGL_DEVICE_ID_MODE=\"$LIBERO_EGL_DEVICE_ID_MODE\" && export __EGL_VENDOR_LIBRARY_FILENAMES=\"${__EGL_VENDOR_LIBRARY_FILENAMES:-}\" && cd \"$ROOT_DIR\" && export EXP_NAME=\"$EXP_NAME\" && export EVAL_PYTHON=\"$EVAL_PYTHON\"" C-m 2>/dev/null
        touch "$marker"
        sleep 1
    }

    # Launch a single task.
    # Pane assignment is handled outside this function.
    launch_task_on_pane() {
        local suite=$1
        local task_id=$2
        local gpu_id=$3
        local pane_info=$4
        local category="${5:-}"
        local physical_gpu
        physical_gpu=$(resolve_physical_gpu "$gpu_id")
        local status_tag
        status_tag=$(task_status_tag "$suite" "$task_id" "$category")
        local status_file="$TASK_STATUS_DIR/${status_tag}.status"
        local suite_result_dir
        suite_result_dir=$(get_suite_result_dir "$suite" "$category")
        local result_file="$suite_result_dir/gpu${physical_gpu}_task${task_id}_results.json"
        local log_file="$TASK_LOG_DIR/${status_tag}_gpu${physical_gpu}.log"
        local category_arg=""
        if [ -n "$category" ]; then
            # Use shell-escaped override to safely carry spaces.
            # e.g. EVALUATION.category_value=Sensor\ Noise
            category_arg="$(printf "%q" "EVALUATION.category_value=$category")"
        fi
        
        rm -f "$status_file"
        # Mark RUNNING before launching so the scheduler can reclaim "ghost" slots if tmux fails
        # to start the worker (otherwise gpu_load/task_gpu_map can get stuck at max forever).
        # Status format: STATUS|gpu_id|meta|ts|log_file
        # - RUNNING: meta is worker pid if known, 0 if unknown
        # - SUCCESS/FAILED: meta is return code
        # Do not pre-create the log file. Creating it eagerly makes many empty
        # log files (and can trigger false "no output" heuristics elsewhere).
        echo "RUNNING|$gpu_id|0|$(date +%s)|$log_file" > "$status_file"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching task: $suite task_id=$task_id category='${category}' on GPU$(gpu_slot_label "$gpu_id") pane $pane_info"
        
        # Launch the task in a tmux pane.
        # IMPORTANT: avoid sending a single huge one-liner (can get truncated / quotes can break),
        # which leads to confusing errors like "-bash: : No such file or directory".
        # Instead, send a few short lines.
        tmux select-pane -t $SESSION_NAME:$pane_info 2>/dev/null
        init_pane_env_once "$pane_info"
        tmux send-keys -t $SESSION_NAME:$pane_info "clear" C-m 2>/dev/null
        tmux send-keys -t $SESSION_NAME:$pane_info "cd \"$ROOT_DIR\"" C-m 2>/dev/null
        tmux send-keys -t $SESSION_NAME:$pane_info "STATUS_FILE='$status_file'; LOG_FILE='$log_file'; RESULT_FILE='$result_file'" C-m 2>/dev/null
        tmux send-keys -t $SESSION_NAME:$pane_info "LOGICAL_GPU='$gpu_id'; PHYSICAL_GPU='$physical_gpu'; SUITE='$suite'; TASK_ID='$task_id'" C-m 2>/dev/null
        # Use the physical GPU label for CUDA so the requested card is selected.
        # eval_libero_single.py remaps MUJOCO_EGL_DEVICE_ID to the process-local EGL ordinal
        # after robosuite's import-time CUDA_VISIBLE_DEVICES assertion has passed.
        tmux send-keys -t $SESSION_NAME:$pane_info \
            "( unset NVIDIA_VISIBLE_DEVICES; export CUDA_VISIBLE_DEVICES=\$PHYSICAL_GPU MUJOCO_EGL_DEVICE_ID=\$PHYSICAL_GPU; echo \"[worker-env] CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=\$MUJOCO_EGL_DEVICE_ID MUJOCO_GL=\${MUJOCO_GL:-} PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-} LIBERO_EGL_DEVICE_ID_MODE=\${LIBERO_EGL_DEVICE_ID_MODE:-} __EGL_VENDOR_LIBRARY_FILENAMES=\${__EGL_VENDOR_LIBRARY_FILENAMES:-} NVIDIA_VISIBLE_DEVICES=\${NVIDIA_VISIBLE_DEVICES:-}\"; \"\$EVAL_PYTHON\" experiments/libero/eval_libero_single.py task=$CONFIG ckpt=$CKPT EVALUATION.task_suite_name=\$SUITE EVALUATION.task_id=\$TASK_ID gpu_id=${physical_gpu} EVALUATION.num_trials=$NUM_TRIALS EVALUATION.output_dir=$OUTPUT_DIR $category_arg $MODEL_ADAPTER_OVERRIDE $MODEL_ENDPOINT_OVERRIDE $DATASET_STATS_OVERRIDE ) > \"\$LOG_FILE\" 2>&1 & pid=\$!; echo \"RUNNING|$gpu_id|\$pid|\$(date +%s)|\$LOG_FILE\" > \"\$STATUS_FILE\"; wait \$pid; rc=\$?; if [ \$rc -eq 0 ] && [ -f \"\$RESULT_FILE\" ]; then echo \"SUCCESS|$gpu_id|\$rc|\$(date +%s)|\$LOG_FILE\" > \"\$STATUS_FILE\"; else echo \"FAILED|$gpu_id|\$rc|\$(date +%s)|\$LOG_FILE\" > \"\$STATUS_FILE\"; fi" \
            C-m 2>/dev/null
        return 0
    }

    # Launch eval_libero_single.py directly from this scheduler (reliable pid + logging).
    launch_task_worker() {
        local suite=$1
        local task_id=$2
        local gpu_id=$3
        local category="${4:-}"
        local physical_gpu
        physical_gpu=$(resolve_physical_gpu "$gpu_id")
        local status_tag
        status_tag=$(task_status_tag "$suite" "$task_id" "$category")
        local status_file="$TASK_STATUS_DIR/${status_tag}.status"
        local suite_result_dir
        suite_result_dir=$(get_suite_result_dir "$suite" "$category")
        local result_file="$suite_result_dir/gpu${physical_gpu}_task${task_id}_results.json"
        local log_file="$TASK_LOG_DIR/${status_tag}_gpu${physical_gpu}.log"

        rm -f "$status_file"

        local -a hydra_args=(
            "task=$CONFIG"
            "ckpt=$CKPT"
            "EVALUATION.task_suite_name=$suite"
            "EVALUATION.task_id=$task_id"
            "gpu_id=${physical_gpu}"
            "EVALUATION.num_trials=$NUM_TRIALS"
            "EVALUATION.output_dir=$OUTPUT_DIR"
        )
        if [ -n "$category" ]; then
            hydra_args+=("EVALUATION.category_value='${category}'")
        fi
        if [ -n "$MODEL_ADAPTER_OVERRIDE" ]; then
            hydra_args+=("$MODEL_ADAPTER_OVERRIDE")
        fi
        if [ -n "$MODEL_ENDPOINT_OVERRIDE" ]; then
            hydra_args+=("$MODEL_ENDPOINT_OVERRIDE")
        fi
        if [ -n "$DATASET_STATS_OVERRIDE" ]; then
            hydra_args+=("$DATASET_STATS_OVERRIDE")
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching task (direct): $suite task_id=$task_id category='${category}' on GPU$(gpu_slot_label "$gpu_id")"

        (
            export PYTHONPATH="$EVAL_PYTHONPATH"
            export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH"
            export MUJOCO_GL="$MUJOCO_GL"
            export PYOPENGL_PLATFORM="$PYOPENGL_PLATFORM"
            export LIBERO_EGL_DEVICE_ID_MODE="$LIBERO_EGL_DEVICE_ID_MODE"
            export MUJOCO_EGL_DEVICE_ID="$physical_gpu"
            unset NVIDIA_VISIBLE_DEVICES
            export EXP_NAME="$EXP_NAME" && export EVAL_PYTHON="$EVAL_PYTHON"
            export WAM_SKIP_LOAD_ENGINE="${WAM_SKIP_LOAD_ENGINE:-1}"
            cd "$ROOT_DIR" || exit 1
            echo "[worker-env] CUDA_VISIBLE_DEVICES=$physical_gpu MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID MUJOCO_GL=${MUJOCO_GL:-} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-} LIBERO_EGL_DEVICE_ID_MODE=${LIBERO_EGL_DEVICE_ID_MODE:-} __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}" > "$log_file"
            CUDA_VISIBLE_DEVICES="$physical_gpu" "$EVAL_PYTHON" experiments/libero/eval_libero_single.py \
                "${hydra_args[@]}" >> "$log_file" 2>&1
            rc=$?
            if [ "$rc" -eq 0 ] && { [ -f "$result_file" ] || task_has_any_result "$suite" "$task_id" "$category"; }; then
                echo "SUCCESS|$gpu_id|$rc|$(date +%s)|$log_file" > "$status_file"
            else
                echo "FAILED|$gpu_id|$rc|$(date +%s)|$log_file" > "$status_file"
            fi
        ) &
        local pid=$!
        echo "RUNNING|$gpu_id|$pid|$(date +%s)|$log_file" > "$status_file"
    }

    launch_task() {
        local suite=$1
        local task_id=$2
        local gpu_id=$3
        local pane_info=$4
        local category="${5:-}"

        record_task_gpu_mapping "$suite" "$task_id" "$gpu_id" "$category"
        local new_load=$(increment_gpu_load "$gpu_id")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Assigned task: $suite task_id=$task_id category='${category}' -> GPU$(gpu_slot_label "$gpu_id") (load: $new_load/$MAX_TASKS_PER_GPU)"
        if [ "$LAUNCH_VIA_TMUX" = "1" ]; then
            launch_task_on_pane "$suite" "$task_id" "$gpu_id" "$pane_info" "$category"
        else
            launch_task_worker "$suite" "$task_id" "$gpu_id" "$category"
        fi
    }
    
    # Worker may be the subshell from launch_task_worker; treat subshell or its eval child as alive.
    worker_pid_is_alive() {
        local pid="${1:-0}"
        [ "$pid" -gt 0 ] 2>/dev/null || return 1
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        if ps -p "$pid" -o args= 2>/dev/null | grep -Fq 'eval_libero_single.py'; then
            return 0
        fi
        pgrep -P "$pid" -f 'eval_libero_single.py' >/dev/null 2>&1
    }

    # Recompute gpu_load from task_gpu_map so it cannot drift (e.g. ps=2 but 0:2+1:2).
    rebuild_gpu_load_from_task_map() {
        local gpu map_line map_gpu load
        for gpu in "${GPU_ARRAY[@]}"; do
            update_gpu_load "$gpu" 0
        done
        if [ ! -f "$TASK_GPU_MAP_FILE" ] || [ ! -s "$TASK_GPU_MAP_FILE" ]; then
            return 0
        fi
        while IFS= read -r map_line; do
            [ -z "$map_line" ] && continue
            map_gpu="${map_line##*:}"
            load=$(get_gpu_load "$map_gpu")
            update_gpu_load "$map_gpu" $((load + 1))
        done < "$TASK_GPU_MAP_FILE"
    }

    # Check completed tasks and clean up finished entries
    cleanup_completed_tasks() {
        CLEANED_COUNT=0
        NEW_FAILURE_COUNT=0
        local launch_stale_sec=${LAUNCH_STALE_SEC:-180}
        local disable_launch_stale_reclaim=${DISABLE_LAUNCH_STALE_RECLAIM:-1}
        local ghost_reclaim_sec=${GHOST_RECLAIM_SEC:-0}

        if [ ! -f "$TASK_GPU_MAP_FILE" ] || [ ! -s "$TASK_GPU_MAP_FILE" ]; then
            rebuild_gpu_load_from_task_map
            return 0
        fi

        local temp_map="$TASK_GPU_MAP_FILE.cleanup"
        > "$temp_map"

        while IFS= read -r map_line; do
            [ -z "$map_line" ] && continue
            local gpu_id="${map_line##*:}"
            local task_info="${map_line%:*}"
            parse_task_fields "$task_info"
            local suite="$PARSED_SUITE"
            local task_id="$PARSED_TASK_ID"
            local category="$PARSED_CATEGORY"
            [ -z "$suite" ] || [ -z "$task_id" ] && continue

            local physical_gpu
            physical_gpu=$(resolve_physical_gpu "$gpu_id")
            local status_tag
            status_tag=$(task_status_tag "$suite" "$task_id" "$category")
            local status_file="$TASK_STATUS_DIR/${status_tag}.status"
            local suite_result_dir
            suite_result_dir=$(get_suite_result_dir "$suite" "$category")
            local result_file="$suite_result_dir/gpu${physical_gpu}_task${task_id}_results.json"
            local log_file="$TASK_LOG_DIR/${status_tag}_gpu${physical_gpu}.log"

            # The result file exists: the task succeeded, so release the mapping and GPU load
            if [ -f "$result_file" ] || task_has_any_result "$suite" "$task_id" "$category"; then
                local new_load=$(decrement_gpu_load "$gpu_id")
                rm -f "$status_file"
                mark_task_completed "$suite" "$task_id" "$category"
                ((CLEANED_COUNT++))
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task completed: $suite task_id=$task_id GPU$gpu_id released (load: $new_load/$MAX_TASKS_PER_GPU)"
                continue
            fi

            # The task process exited with failure: detect it, report it, and reclaim the mapping
            if [ -f "$status_file" ]; then
                IFS='|' read -r status status_gpu status_meta status_ts status_log < "$status_file"
                if [ "$status" = "RUNNING" ]; then
                    # RUNNING meta is the worker pid if known; 0 means unknown (e.g. tmux didn't start).
                    # Prefer pid-based liveness checks; fall back to stale-launch heuristics when pid is unknown.
                    local pid="${status_meta:-0}"
                    if [ "$pid" -gt 0 ] 2>/dev/null; then
                        if ! worker_pid_is_alive "$pid"; then
                            # Subshell/eval gone and no result => reclaim ghost slot.
                            local new_load=$(decrement_gpu_load "$gpu_id")
                            mark_task_failed "$suite" "$task_id" "$gpu_id" "pid_gone" "${status_log:-$log_file}" "$category"
                            enqueue_retry_task "$suite" "$task_id" "$category"
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task worker gone: $suite task_id=$task_id pid=$pid GPU$gpu_id reclaimed (load: $new_load/$MAX_TASKS_PER_GPU)"
                            rm -f "$status_file"
                            continue
                        fi
                    fi

                    # Optional "launch stale" reclaim:
                    # Historically we reclaimed slots when a task stayed RUNNING for too long with
                    # zero log output (often due to tmux/env init hiccups). In practice this can
                    # misclassify slow startups and cause duplicate task runs, so it is disabled
                    # by default. Enable by setting DISABLE_LAUNCH_STALE_RECLAIM=0.
                    local now_ts
                    now_ts=$(date +%s)
                    local age=$((now_ts - ${status_ts:-now_ts}))
                    local log_size=0
                    if [ -f "$log_file" ]; then
                        log_size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
                    fi
                    # If tmux didn't manage to start the worker, pid can stay 0 and the log file
                    # may never appear. Reclaim such "ghost launches" so the scheduler doesn't
                    # deadlock at max load. This does NOT enqueue retries.
                    if [ "${pid:-0}" -eq 0 ] 2>/dev/null && [ "$ghost_reclaim_sec" -gt 0 ] 2>/dev/null; then
                        if [ "$age" -ge "$ghost_reclaim_sec" ] && [ ! -f "$log_file" ]; then
                            local new_load=$(decrement_gpu_load "$gpu_id")
                            mark_task_failed "$suite" "$task_id" "$gpu_id" "ghost_launch" "${status_log:-$log_file}" "$category"
                            enqueue_retry_task "$suite" "$task_id" "$category"
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ghost launch reclaimed: $suite task_id=$task_id GPU$gpu_id reclaimed (load: $new_load/$MAX_TASKS_PER_GPU)"
                            rm -f "$status_file"
                            continue
                        fi
                    fi
                    if [ "${disable_launch_stale_reclaim}" = "0" ] && [ "$launch_stale_sec" -gt 0 ] 2>/dev/null; then
                        if [ "$age" -ge "$launch_stale_sec" ] && [ "$log_size" -eq 0 ]; then
                            local new_load=$(decrement_gpu_load "$gpu_id")
                            mark_task_failed "$suite" "$task_id" "$gpu_id" "launch_stale" "${status_log:-$log_file}" "$category"
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task launch stale: $suite task_id=$task_id GPU$gpu_id reclaimed (load: $new_load/$MAX_TASKS_PER_GPU)"
                            rm -f "$status_file"
                            continue
                        fi
                    fi
                fi
                if [ "$status" = "FAILED" ]; then
                    local new_load=$(decrement_gpu_load "$gpu_id")
                    mark_task_failed "$suite" "$task_id" "$gpu_id" "${status_meta:-unknown}" "${status_log:-unknown}" "$category"
                    ((NEW_FAILURE_COUNT++))
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task failed: $suite task_id=$task_id rc=$status_meta GPU$gpu_id (current load: $new_load/$MAX_TASKS_PER_GPU)"
                    rm -f "$status_file"
                    continue
                fi
                if [ "$status" = "SUCCESS" ]; then
                    local new_load=$(decrement_gpu_load "$gpu_id")
                    rm -f "$status_file"
                    mark_task_completed "$suite" "$task_id" "$category"
                    ((CLEANED_COUNT++))
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task completed (status file): $suite task_id=$task_id GPU$gpu_id released (load: $new_load/$MAX_TASKS_PER_GPU)"
                    continue
                fi
            fi

            # Still running: keep the mapping
            echo "${task_info}:${gpu_id}" >> "$temp_map"
        done < "$TASK_GPU_MAP_FILE"

        mv "$temp_map" "$TASK_GPU_MAP_FILE"
        rebuild_gpu_load_from_task_map
        return 0
    }

    # Incremental completion tracking (avoids per-round ls over the full pending queue).
    COMPLETED_KEYS_FILE="$OUTPUT_DIR/completed_task_keys.txt"
    PENDING_CURSOR_FILE="$OUTPUT_DIR/pending_cursor"
    RETRY_TASKS_FILE="$OUTPUT_DIR/retry_tasks.txt"
    # O(1) membership checks; avoids grep on a growing completed_task_keys.txt every second.
    declare -A COMPLETED_KEYS_MAP=()

    load_completed_keys_map() {
        COMPLETED_KEYS_MAP=()
        if [ ! -f "$COMPLETED_KEYS_FILE" ]; then
            return 0
        fi
        local key
        while IFS= read -r key; do
            [ -z "$key" ] && continue
            COMPLETED_KEYS_MAP["$key"]=1
        done < "$COMPLETED_KEYS_FILE"
    }

    enqueue_retry_task() {
        local suite="$1"
        local task_id="$2"
        local category="${3:-}"
        local task_key
        task_key=$(make_task_key "$suite" "$task_id" "$category")
        if task_key_is_completed "$suite" "$task_id" "$category"; then
            return 0
        fi
        # Avoid duplicate retries.
        if [ -f "$RETRY_TASKS_FILE" ] && grep -qxF "$task_key" "$RETRY_TASKS_FILE" 2>/dev/null; then
            return 0
        fi
        echo "$task_key" >> "$RETRY_TASKS_FILE"
    }

    pop_retry_task_key() {
        if [ ! -s "$RETRY_TASKS_FILE" ]; then
            return 1
        fi
        local first
        first=$(sed -n '1p' "$RETRY_TASKS_FILE" 2>/dev/null || true)
        if [ -z "$first" ]; then
            return 1
        fi
        # Remove first line atomically.
        local tmp="$RETRY_TASKS_FILE.tmp"
        sed '1d' "$RETRY_TASKS_FILE" > "$tmp" 2>/dev/null || true
        mv "$tmp" "$RETRY_TASKS_FILE"
        echo "$first"
        return 0
    }

    prepend_retry_task_key() {
        local key="$1"
        local tmp="$RETRY_TASKS_FILE.tmp"
        {
            echo "$key"
            [ -f "$RETRY_TASKS_FILE" ] && cat "$RETRY_TASKS_FILE"
        } > "$tmp"
        mv "$tmp" "$RETRY_TASKS_FILE"
    }

    task_key_is_completed() {
        local suite="$1"
        local task_id="$2"
        local category="${3:-}"
        local task_key
        task_key=$(make_task_key "$suite" "$task_id" "$category")
        [ -n "${COMPLETED_KEYS_MAP[$task_key]:-}" ]
    }

    mark_task_completed() {
        local suite="$1"
        local task_id="$2"
        local category="${3:-}"
        local task_key
        task_key=$(make_task_key "$suite" "$task_id" "$category")
        if [ -z "${COMPLETED_KEYS_MAP[$task_key]:-}" ]; then
            COMPLETED_KEYS_MAP["$task_key"]=1
            echo "$task_key" >> "$COMPLETED_KEYS_FILE"
            batch_completed=$((batch_completed + 1))
            if [ "${SKIP_CURSOR_SYNC:-0}" != "1" ]; then
                advance_pending_cursor
            fi
        fi
    }

    task_has_any_result() {
        local suite="$1"
        local task_id="$2"
        local category="${3:-}"
        local suite_result_dir
        suite_result_dir=$(get_suite_result_dir "$suite" "$category")
        compgen -G "$suite_result_dir/gpu*_task${task_id}_results.json" >/dev/null 2>&1
    }

    # Scan task list and merge any existing results into completed_task_keys.txt (resume-safe).
    init_completed_from_task_file() {
        local list_file="$1"
        local line
        touch "$COMPLETED_KEYS_FILE"
        batch_completed=$(wc -l < "$COMPLETED_KEYS_FILE" 2>/dev/null | tr -d ' ')
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            parse_task_fields "$line"
            if task_has_any_result "$PARSED_SUITE" "$PARSED_TASK_ID" "$PARSED_CATEGORY"; then
                mark_task_completed "$PARSED_SUITE" "$PARSED_TASK_ID" "$PARSED_CATEGORY"
            fi
        done < "$list_file"
    }

    # First task index (0-based) in pending_tasks.txt that still needs to run.
    find_first_pending_cursor() {
        local list_file="$1"
        local line_idx=0
        local line
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            parse_task_fields "$line"
            if ! task_key_is_completed "$PARSED_SUITE" "$PARSED_TASK_ID" "$PARSED_CATEGORY"; then
                echo "$line_idx"
                return 0
            fi
            line_idx=$((line_idx + 1))
        done < "$list_file"
        echo "$line_idx"
    }

    # Move pending_cursor forward across consecutive completed lines only (cheap hot path).
    advance_pending_cursor() {
        if [ -z "${PENDING_TASKS_FILE:-}" ] || [ ! -f "$PENDING_TASKS_FILE" ]; then
            return 0
        fi
        local pos total_lines line
        pos=$(read_pending_cursor)
        total_lines=$(get_pending_line_count)
        while [ "$pos" -lt "$total_lines" ]; do
            line=$(sed -n "$((pos + 1))p" "$PENDING_TASKS_FILE")
            [ -z "$line" ] && break
            parse_task_fields "$line"
            if ! task_key_is_completed "$PARSED_SUITE" "$PARSED_TASK_ID" "$PARSED_CATEGORY"; then
                write_pending_cursor "$pos"
                return 0
            fi
            pos=$((pos + 1))
        done
        write_pending_cursor "$pos"
    }

    # Full rescan from line 0 — use only at startup/resume, not every monitor tick.
    sync_pending_cursor() {
        if [ -z "${PENDING_TASKS_FILE:-}" ] || [ ! -f "$PENDING_TASKS_FILE" ]; then
            return 0
        fi
        local new_cursor
        new_cursor=$(find_first_pending_cursor "$PENDING_TASKS_FILE")
        write_pending_cursor "$new_cursor"
    }

    get_scheduler_capacity() {
        echo $((${#GPU_ARRAY[@]} * MAX_TASKS_PER_GPU))
    }

    # Slot accounting uses gpu_load (incremented/decremented on launch/cleanup), not task_gpu_map
    # line count — map can retain stale rows and block launches while only 2 eval are alive.
    get_total_gpu_load() {
        local total=0
        for gpu in "${GPU_ARRAY[@]}"; do
            local load
            load=$(get_gpu_load "$gpu")
            total=$((total + load))
        done
        echo "$total"
    }

    get_running_task_count() {
        get_total_gpu_load
    }

    get_task_map_count() {
        if [ ! -f "$TASK_GPU_MAP_FILE" ]; then
            echo 0
            return 0
        fi
        wc -l < "$TASK_GPU_MAP_FILE" 2>/dev/null | tr -d ' '
    }

    get_slots_available() {
        local capacity used
        capacity=$(get_scheduler_capacity)
        used=$(get_total_gpu_load)
        local free=$((capacity - used))
        [ "$free" -lt 0 ] && free=0
        echo "$free"
    }

    read_pending_cursor() {
        if [ -f "$PENDING_CURSOR_FILE" ]; then
            tr -d ' \n' < "$PENDING_CURSOR_FILE"
        else
            echo 0
        fi
    }

    write_pending_cursor() {
        echo "$1" > "$PENDING_CURSOR_FILE"
    }

    get_pending_line_count() {
        if [ ! -f "$PENDING_TASKS_FILE" ]; then
            echo 0
            return 0
        fi
        wc -l < "$PENDING_TASKS_FILE" 2>/dev/null | tr -d ' '
    }

    get_pending_queue_count() {
        # Remaining work = total tasks minus those with results recorded.
        echo $((total_tasks - batch_completed))
    }

    # Launch tasks from pending queue.
    # STRICT_SEQUENTIAL=0 (default): parallel slot pool — up to NUM_GPUS*MAX_TASKS_PER_GPU eval workers.
    #   Each free slot takes the next incomplete task in pending_tasks.txt order (pending_cursor = first incomplete).
    #   A slot finishes -> cleanup frees GPU load -> next round assigns another task to that GPU.
    # STRICT_SEQUENTIAL=1: only one eval process globally (debug / reproduce issues).
    launch_tasks_from_pending_queue() {
        local slots="$1"
        local limit="$2"
        local to_launch="$slots"
        if [ "$to_launch" -gt "$limit" ]; then
            to_launch="$limit"
        fi
        if [ "$to_launch" -le 0 ]; then
            return 0
        fi

        advance_pending_cursor

        if [ "${STRICT_SEQUENTIAL:-0}" = "1" ]; then
            if [ "$(get_running_task_count)" -gt 0 ]; then
                echo 0
                return 0
            fi
            to_launch=1
        fi

        local scan_pos total_lines launched=0
        scan_pos=$(read_pending_cursor)
        total_lines=$(get_pending_line_count)

        while [ "$launched" -lt "$to_launch" ]; do
            local pending_line from_retry=0

            if [ -s "$RETRY_TASKS_FILE" ]; then
                pending_line=$(pop_retry_task_key || true)
                if [ -n "$pending_line" ]; then
                    from_retry=1
                fi
            fi

            if [ "$from_retry" -eq 0 ]; then
                if [ "$scan_pos" -ge "$total_lines" ]; then
                    break
                fi
                pending_line=$(sed -n "$((scan_pos + 1))p" "$PENDING_TASKS_FILE")
                [ -z "$pending_line" ] && break
            fi

            parse_task_fields "$pending_line"
            local suite="$PARSED_SUITE"
            local task_id="$PARSED_TASK_ID"
            local category="$PARSED_CATEGORY"

            if task_key_is_completed "$suite" "$task_id" "$category"; then
                scan_pos=$((scan_pos + 1))
                continue
            fi

            if [ -n "$(get_task_gpu "$suite" "$task_id" "$category")" ]; then
                if [ "${STRICT_SEQUENTIAL:-0}" = "1" ]; then
                    break
                fi
                # Parallel: this task is already on a slot; try next line in the list this round.
                scan_pos=$((scan_pos + 1))
                continue
            fi

            local gpu_id
            gpu_id=$(find_least_loaded_gpu)
            if [ -z "$gpu_id" ]; then
                if [ "$from_retry" -eq 1 ]; then
                    prepend_retry_task_key "$pending_line"
                fi
                break
            fi

            local pane_info="0.0"
            if [ "$LAUNCH_VIA_TMUX" = "1" ]; then
                local window_id pane_id
                window_id=$((NEXT_PANE_INDEX / MAX_PANES))
                pane_id=$((NEXT_PANE_INDEX % MAX_PANES))
                pane_info="$window_id.$pane_id"
                ensure_pane_exists "$window_id" "$pane_id"
                NEXT_PANE_INDEX=$((NEXT_PANE_INDEX + 1))
            fi

            launch_task "$suite" "$task_id" "$gpu_id" "$pane_info" "$category"
            launched=$((launched + 1))

            if [ "${STRICT_SEQUENTIAL:-0}" = "1" ]; then
                break
            fi
            if [ "$from_retry" -eq 0 ]; then
                scan_pos=$((scan_pos + 1))
            fi
        done

        advance_pending_cursor
        echo "$launched"
    }

    
    # Main loop for dynamic task scheduling.
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting dynamic task scheduling..."
    
    PENDING_TASKS_FILE="$OUTPUT_DIR/pending_tasks.txt"
    cp "$task_list_file" "$PENDING_TASKS_FILE"

    # Drop stale in-memory scheduler state from a previous crashed run in the same output dir.
    > "$TASK_GPU_MAP_FILE"
    init_gpu_load_tracking
    
    local total_tasks=$(wc -l < "$task_list_file")
    local monitoring_interval=${MONITORING_INTERVAL:-10}
    local last_status_time=0
    local status_interval=${STATUS_INTERVAL:-30}
    local max_launch_per_round=${MAX_LAUNCH_PER_ROUND:-$((NUM_GPUS * MAX_TASKS_PER_GPU))}
    local batch_completed=0
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total tasks: $total_tasks"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max tasks per GPU: $MAX_TASKS_PER_GPU"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max launch per round: $max_launch_per_round"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Available GPUs: ${GPU_ARRAY[*]}"
    local scheduler_capacity
    scheduler_capacity=$(get_scheduler_capacity)
    if [ "${STRICT_SEQUENTIAL:-0}" = "1" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scheduler: STRICT_SEQUENTIAL=1 (single eval worker)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scheduler: parallel slot pool capacity=$scheduler_capacity (${#GPU_ARRAY[@]} GPUs x MAX_TASKS_PER_GPU=$MAX_TASKS_PER_GPU); slot freed -> next task in order"
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scanning existing results (merge into completed_task_keys)..."
    load_completed_keys_map
    SKIP_CURSOR_SYNC=1
    init_completed_from_task_file "$task_list_file"
    unset SKIP_CURSOR_SYNC
    sync_pending_cursor
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch completed (after scan): $batch_completed/$total_tasks, pending_cursor=$(read_pending_cursor)"
    local max_initial_tasks
    max_initial_tasks=$(get_scheduler_capacity)
    if [ "${STRICT_SEQUENTIAL:-0}" = "1" ]; then
        max_initial_tasks=1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting the initial launch phase (up to $max_initial_tasks tasks)..."
    local initial_launched
    initial_launched=$(launch_tasks_from_pending_queue "$max_initial_tasks" "$max_initial_tasks")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initial launch completed, started $initial_launched tasks"
    
    while true; do
        current_time=$(date +%s)

        cleanup_completed_tasks
        cleaned=$CLEANED_COUNT
        if [ "$cleaned" -gt 0 ] 2>/dev/null; then
            advance_pending_cursor
        fi
        new_failures=$NEW_FAILURE_COUNT
        total_failed=$(wc -l < "$FAILED_TASKS_FILE" 2>/dev/null || echo 0)

        # By default, do NOT stop the whole scheduler on a single subtask failure.
        # Set STOP_ON_FAILURE=1 to restore the old "fail-fast" behavior.
        if [ "$new_failures" -gt 0 ]; then
            if [ "${STOP_ON_FAILURE:-0}" = "1" ]; then
                echo "Detected failed subtasks, stopping the scheduler. Failure details: $FAILED_TASKS_FILE"
                cat "$FAILED_TASKS_FILE"
                return 2
            fi
            echo "Detected failed subtasks, continuing (STOP_ON_FAILURE!=1). Failure details: $FAILED_TASKS_FILE"
            tail -n 20 "$FAILED_TASKS_FILE" 2>/dev/null || true
        fi

        total_completed=$batch_completed
        if [ "$total_completed" -eq "$total_tasks" ]; then
            echo "All tasks are complete!"
            break
        fi

        launched_this_round=0
        local slots_available
        slots_available=$(get_slots_available)
        if [ "$slots_available" -gt 0 ]; then
            launched_this_round=$(launch_tasks_from_pending_queue "$slots_available" "$max_launch_per_round")
        fi

        running_count=$(get_running_task_count)
        pending_count=$(get_pending_queue_count)
        local cursor_at_end=0
        if [ "$(read_pending_cursor)" -ge "$(get_pending_line_count)" ]; then
            cursor_at_end=1
        fi

        if [ "$running_count" -eq 0 ] && [ "$cursor_at_end" -eq 1 ] && [ "$total_completed" -lt "$total_tasks" ]; then
            echo "Scheduling inconsistency: no running tasks and no pending tasks, but not all tasks are complete."
            echo "Completed: $total_completed/$total_tasks, failed: $total_failed"
            [ -s "$FAILED_TASKS_FILE" ] && cat "$FAILED_TASKS_FILE"
            return 2
        fi
        
        if [ $((current_time - last_status_time)) -ge $status_interval ]; then
            local cur_line
            cur_line=$(sed -n "$(( $(read_pending_cursor) + 1 ))p" "$PENDING_TASKS_FILE" 2>/dev/null || true)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Scheduling Status $(date '+%H:%M:%S') ==="
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total tasks: $total_tasks"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $total_completed"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remaining: $((total_tasks - total_completed))"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] pending_cursor=$(read_pending_cursor) next_line=${cur_line:-'(done)'}"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running (gpu_load): $running_count"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] task_gpu_map lines: $(get_task_map_count) (stale lines block launch if >> gpu_load)"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pending: $pending_count"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed: $total_failed"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched this round: $launched_this_round"
            if [ "$cleaned" -gt 0 ] 2>/dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaned this round: $cleaned"
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] === GPU Load Status ==="
            local physical
            for gpu in "${GPU_ARRAY[@]}"; do
                load=$(get_gpu_load $gpu)
                percentage=$((load * 100 / MAX_TASKS_PER_GPU))
                physical=$(resolve_physical_gpu "$gpu")
                if [ "$physical" != "$gpu" ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $physical (slot $gpu): $load/$MAX_TASKS_PER_GPU tasks ($percentage%)"
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: $load/$MAX_TASKS_PER_GPU tasks ($percentage%)"
                fi
            done
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================="
            
            show_debug_info
            echo ""
            last_status_time=$current_time
        fi
        
        sleep $monitoring_interval
    done
    
    rm -f "$PENDING_TASKS_FILE" "$PENDING_TASKS_FILE.processing" "$PENDING_CURSOR_FILE"

    # Check the final result
    echo "All tasks completed successfully!"
    # Run the result summarization script
    echo "Generating evaluation report..."
    "$EVAL_PYTHON" experiments/libero/summarize_results.py --output_dir="$OUTPUT_DIR"
}


# Entrypoint
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Check whether a task file argument is provided
    if [ $# -lt 1 ]; then
        echo "Error: task file path is required"
        echo "Usage: $0 <task_file>"
        exit 1
    fi
    test_file="$1"
    run_libero_eval "$test_file"
    exit $?
fi
