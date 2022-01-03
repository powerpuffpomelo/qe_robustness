#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_mlm \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/pretrain_generator_ende.yaml" \
    --log_path "./log" \
    --saveto "./save/" \
    --use_gpu
