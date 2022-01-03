#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.generate_dev \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/train_directqe_enzh/generate_pseudo_enzh.yaml" \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/generator/generator_mlm.best.final" \
    --log_path "/home/yanym/data/wmt20_enzh_pseudo/" \
    --saveto "/home/yanym/data/wmt20_enzh_pseudo/" \
    --use_gpu

# bash scripts/generate_dev.sh