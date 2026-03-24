#!/bin/bash
CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_FILE=${1:-"${CURR_FILE_DIR}/example_t2v.json"}
CONFIG_FILE=${2:-"${CURR_FILE_DIR}/configs/kairos_4b_config_DMD_A800.py"}
GPU=${3:-"4"}

CODE_DIR="$(cd "${CURR_FILE_DIR}/../../.." && pwd)"


cd $CODE_DIR

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH
python ${CODE_DIR}/kairos/third_party/manage_libs.py

# NOTE:
# - This script supports **2-GPU** and **4-GPU** parallel inference only (nproc-per-node=2 or 4).
# - When **CFG-parallel** is enabled (cond/uncond split), it runs as **two** parallel groups:
#     - 2 groups × 2 GPUs (requires **4 GPUs** total), or
#     - 2 groups × 4 GPUs (requires **8 GPUs** total).
# - Other configurations (e.g., 6 GPUs, multi-node) are not supported yet.

torchrun --nnodes=1  --master_port 29556 --nproc-per-node=$GPU \
    ${CURR_FILE_DIR}/inference.py \
        --input_file ${INPUT_FILE} \
        --config_file ${CONFIG_FILE}


