#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_lm \
    --model_name "generator_lm_backward" \
    --lm LM_backward \
    --config_path "./configs/pretrain_generator_ende.yaml" \
    --log_path "./log" \
    --saveto "./save/" \
    --use_gpu
