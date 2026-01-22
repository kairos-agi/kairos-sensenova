#!/bin/bash

INPUT_FILE=${1:-"examples/kairos/example_t2v.json"}

CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${CURR_FILE_DIR}/.." && pwd)"


cd $CODE_DIR

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH


torchrun --nnodes=1  --master_port 29556 --nproc-per-node=1 \
    ${CODE_DIR}/examples/inference.py \
        --input_file ${INPUT_FILE} \


