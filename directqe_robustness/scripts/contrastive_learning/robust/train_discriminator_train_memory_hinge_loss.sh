#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator_one_class \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_lm_forward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_forward.best.final" \
    --pretrain_generator_lm_backward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_backward.best.final" \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/generator/generator_mlm.best.final" \
    --config_path "./configs/train_directqe_enzh/pretrain_discriminator_enzh_one_class.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_pretrained_256_train_memory_hinge_loss" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_pretrained_256_train_memory_hinge_loss/" \
    --criterion "hinge_loss" \
    --use_gpu

# bash scripts/contrastive_learning/robust/train_discriminator_train_memory_hinge_loss.sh