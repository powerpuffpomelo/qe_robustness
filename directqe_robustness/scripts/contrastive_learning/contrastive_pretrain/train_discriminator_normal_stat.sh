#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator_no_need_2_generate \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/generator/generator_mlm.best.final" \
    --config_path "./configs/train_directqe_enzh/pretrain_discriminator_enzh_stat.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_pretrained_256_normal_stat" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_pretrained_256_normal_stat/" \
    --use_gpu


# bash scripts/contrastive_learning/contrastive_pretrain/train_discriminator_normal_stat.sh