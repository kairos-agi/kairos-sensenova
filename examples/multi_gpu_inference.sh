#!/bin/bash

INPUT_FILE=${1:-"examples/example_t2v.json"}

CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${CURR_FILE_DIR}/.." && pwd)"


cd $CODE_DIR

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH

# NOTE:
# - This script currently supports **4-GPU** parallel inference only (nproc-per-node=4).
# - Other GPU counts / parallel configurations (e.g., 6/8 GPUs, multi-node) will be added in future updates.

GPU=4
torchrun --nnodes=1  --master_port 29556 --nproc-per-node=$GPU \
    ${CODE_DIR}/examples/inference.py \
        --input_file ${INPUT_FILE} \


