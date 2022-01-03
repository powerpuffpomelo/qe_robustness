#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_mlm \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/train_directqe_enzh/pretrain_generator_enzh.yaml" \
    --log_path "/home/yanym/save/directqe_enzh/generator" \
    --saveto "/home/yanym/save/directqe_enzh/generator/" \
    --use_gpu


# bash scripts/train_directqe_enzh/train_generator_mlm_enzh.sh