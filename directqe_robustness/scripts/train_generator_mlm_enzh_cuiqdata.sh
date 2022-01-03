#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_mlm \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/pretrain_generator_enzh_cuiqdata.yaml" \
    --log_path "./log" \
    --saveto "./save_generator_cuiqdata/" \
    --use_gpu


# bash scripts/train_generator_mlm_enzh_cuiqdata.sh > logs/train_generator_mlm_enzh_cuiqdata.log