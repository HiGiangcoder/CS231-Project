#!/bin/bash

export DATA_ROOT="../../dataset/RAF"
export LABEL_FILE="../../dataset/list_patition_label.txt"

mkdir -p logs

python train.py \
    > logs/train_$(date +%Y%m%d_%H%M%S).log \
    2>&1