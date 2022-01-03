#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.generate_dev \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/train_directqe_ende/generate_pseudo_ende.yaml" \
    --pretrain_path "/home/data_ti4_c/yanym/save/ende/generator_temp/generator_mlm.best.final" \
    --log_path "/home/yanym/save/ende/generate_dev" \
    --saveto "/home/yanym/save/ende/generate_dev/" \
    --use_gpu

# bash scripts/train_directqe_ende/generate_dev.sh