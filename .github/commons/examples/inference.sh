#!/bin/bash
CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_FILE=${1:-"${CURR_FILE_DIR}/example_t2v.json"}
CONFIG_FILE=${2:-"${CURR_FILE_DIR}/configs/kairos_4b_config_DMD_A800.py"}

CODE_DIR="$(cd "${CURR_FILE_DIR}/../../.." && pwd)"

cd $CODE_DIR

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH
python ${CODE_DIR}/kairos/third_party/manage_libs.py

torchrun --nnodes=1  --master_port 29556 --nproc-per-node=1 \
    ${CURR_FILE_DIR}/inference.py \
        --input_file ${INPUT_FILE} \
        --config_file ${CONFIG_FILE}


