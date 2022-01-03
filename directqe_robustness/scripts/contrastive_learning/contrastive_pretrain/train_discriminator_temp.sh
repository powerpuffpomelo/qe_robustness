#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator_no_need_2_generate \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/enzh/generator/generator_mlm.best.final" \
    --pretrain_discriminator_path "/data1/yanym/data/toy_contrastive/discriminator_toy2/discriminator.best.ckpt.210" \
    --config_path "./configs/contrastive/pretrain_discriminator_enzh_temp.yaml" \
    --log_path "/data1/yanym/data/toy_contrastive/discriminator_toy2" \
    --saveto "/data1/yanym/data/toy_contrastive/discriminator_toy2/" \
    --use_gpu

# bash scripts/contrastive_learning/contrastive_pretrain/train_discriminator_temp.sh