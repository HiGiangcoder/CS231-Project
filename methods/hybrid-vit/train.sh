#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TRAIN_ROOT="../../dataset/RAF"
export LABEL_FILE="../../dataset/list_patition_label.txt"

python train.py