#!/bin/bash

INPUT_FILE=${1:-"examples/example_t2v_480P.json"}
CONFIG_FILE=${2:-"kairos/configs/kairos_4b_config_DMD.py"}
GPU=${3:-4}

CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${CURR_FILE_DIR}/.." && pwd)"

cd "$CODE_DIR" || exit 1

export PYTHONPATH="${CODE_DIR}:$PYTHONPATH"
python "${CODE_DIR}/kairos/third_party/manage_libs.py"

# NOTE:
# - This script supports 2-GPU, 4-GPU, and 8-GPU parallel inference only.
# - Other configurations (e.g., 6 GPUs, multi-node) are not supported yet.
# - When CFG-parallel is enabled (cond/uncond split), it runs as two parallel groups:
#     - 2 groups × 2 GPUs (requires 4 GPUs total), or
#     - 2 groups × 4 GPUs (requires 8 GPUs total).

if [[ "$GPU" != "2" && "$GPU" != "4" && "$GPU" != "8" ]]; then
    echo "Error: GPU must be 2, 4, or 8"
    exit 1
fi

torchrun --nnodes=1 --master_port 29556 --nproc-per-node="$GPU" \
    "${CODE_DIR}/examples/inference.py" \
    --input_file "$INPUT_FILE" \
    --config_file "$CONFIG_FILE"
