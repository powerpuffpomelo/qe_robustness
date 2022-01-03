#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_mlm \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/train_directqe_ende/pretrain_generator_ende.yaml" \
    --log_path "/home/yanym/save/ende/generator" \
    --saveto "/home/yanym/save/ende/generator" \
    --use_gpu


# bash scripts/train_directqe_ende/train_generator_mlm_ende.sh