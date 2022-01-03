#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_lm_forward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_forward.best.final" \
    --pretrain_generator_lm_backward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_backward.best.final" \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/ende/generator_temp/generator_mlm.best.final" \
    --config_path "./configs/train_directqe_ende/pretrain_discriminator_ende.yaml" \
    --log_path "/home/yanym/save/ende/discriminator_pretrained_256_syn" \
    --saveto "/home/yanym/save/ende/discriminator_pretrained_256_syn/" \
    --use_gpu

# bash scripts/train_directqe_ende/train_discriminator_ende_syn.sh