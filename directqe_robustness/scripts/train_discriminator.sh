#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_lm_forward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_forward.best.final" \
    --pretrain_generator_lm_backward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_backward.best.final" \
    --pretrain_generator_mlm_path "./save_generator_cuiqdata/generator_mlm.best.final" \
    --config_path "./configs/pretrain_discriminator_enzh.yaml" \
    --log_path "./log" \
    --saveto "./save_discriminator_enzh/" \
    --use_gpu

# bash scripts/train_discriminator.sh > logs/train_discriminator_enzh.log