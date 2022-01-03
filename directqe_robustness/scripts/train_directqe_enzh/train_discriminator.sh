#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_lm_forward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_forward.best.final" \
    --pretrain_generator_lm_backward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_backward.best.final" \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/directqe_enzh/generator/generator_mlm.best.final" \
    --config_path "./configs/train_directqe_enzh/pretrain_discriminator_enzh.yaml" \
    --log_path "/home/yanym/save/directqe_enzh/discriminator_pretrained_256_new" \
    --saveto "/home/yanym/save/directqe_enzh/discriminator_pretrained_256_new/" \
    --use_gpu

# bash scripts/train_directqe_enzh/train_discriminator.sh