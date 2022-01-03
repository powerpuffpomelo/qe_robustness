#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.generate_dev \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/train_directqe_enzh/generate_dev_enzh.yaml" \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh/generator/generator_mlm.best.final" \
    --log_path "/home/yanym/save/directqe_enzh/generate_dev" \
    --saveto "/home/yanym/save/directqe_enzh/generate_dev/" \
    --use_gpu

# bash scripts/train_directqe_enzh/generate_dev.sh