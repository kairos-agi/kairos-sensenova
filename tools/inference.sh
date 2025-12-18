#!/bin/bash


CONFIG="${1:-configs/kairos_4b/kairos_4b_config.py}"
WEIGHT_PATH=${2:-"none"}
INPUT_FILE=${3:-"examples/kairos/example_t2v.json"}
OUTPUT_PATH=${4:-"output/t2v"}

CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${CURR_FILE_DIR}/.." && pwd)"


cd $CODE_DIR

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH


torchrun --nnodes=1  --master_port 29556 --nproc-per-node=1 \
    ${CODE_DIR}/tools/inference.py \
        --config ${CONFIG} \
        --checkpoint ${WEIGHT_PATH} \
        --input_file ${INPUT_FILE} \
        --output_dir ${OUTPUT_PATH}


