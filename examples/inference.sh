#!/bin/bash

INPUT_FILE=${1:-"examples/example_t2v_480P.json"}
CONFIG_FILE=${2:-"kairos/configs/kairos_4b_config_DMD.py"}

CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${CURR_FILE_DIR}/.." && pwd)"

cd $CODE_DIR || exit 1

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH
python ${CODE_DIR}/kairos/third_party/manage_libs.py

torchrun --nnodes=1  --master_port 29556 --nproc-per-node=1 \
    ${CODE_DIR}/examples/inference.py \
        --input_file ${INPUT_FILE} \
        --config_file "$CONFIG_FILE"