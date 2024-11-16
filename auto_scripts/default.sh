#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DRAG_DIR=$(dirname $SCRIPT_DIR)

cd $DRAG_DIR
nohup python drag.py \
    --config="./config/drag.yaml" \
    --config="./config/model/llama32_3b.yaml" \
    --config="./config/dataset/trivia_qa.yaml" \
    --log_level=INFO > $DRAG_DIR/script.log 2>&1 &

