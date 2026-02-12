#!/bin/bash

INPUT_FILE=${1:-"examples/example_t2v.json"}

CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${CURR_FILE_DIR}/.." && pwd)"


cd $CODE_DIR

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH
python ${CODE_DIR}/kairos/third_party/manage_libs.py

# NOTE:
# - This script supports **2-GPU** and **4-GPU** parallel inference only (nproc-per-node=2 or 4).
# - When **CFG-parallel** is enabled (cond/uncond split), it runs as **two** parallel groups:
#     - 2 groups × 2 GPUs (requires **4 GPUs** total), or
#     - 2 groups × 4 GPUs (requires **8 GPUs** total).
# - Other configurations (e.g., 6 GPUs, multi-node) are not supported yet.

GPU=4
torchrun --nnodes=1  --master_port 29556 --nproc-per-node=$GPU \
    ${CODE_DIR}/examples/inference.py \
        --input_file ${INPUT_FILE} \


