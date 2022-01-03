#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.generate_dev \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/contrastive/generate_dev_enzh_temp.yaml" \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/generator/generator_mlm.best.final" \
    --log_path "/data1/yanym/data/toy_contrastive/temp/generate_dev" \
    --saveto "/data1/yanym/data/toy_contrastive/temp/generate_dev/" \
    --use_gpu

# bash scripts/contrastive_learning/contrastive_pretrain/generate_dev_temp.sh