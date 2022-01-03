#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.generate_dev_with_parallel \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/train_directqe_ende/generate_dev_ende.yaml" \
    --pretrain_path "/home/data_ti4_c/yanym/save/ende/generator/generator_mlm.best.final" \
    --log_path "/home/yanym/data/wmt19_ende_generate_parallel_pseudo_label" \
    --saveto "/home/yanym/data/wmt19_ende_generate_parallel_pseudo_label/" \
    --split "train" \
    --use_gpu

# bash scripts/train_directqe_enzh/generate_dev_with_parallel.sh