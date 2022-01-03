#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator_no_need_2_generate \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generator_wwm/generator_mlm.best.final" \
    --config_path "./configs/contrastive/pretrain_discriminator_enzh_asyn.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_pretrained_256_asyn" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_pretrained_256_asyn/" \
    --use_gpu


# bash scripts/contrastive_learning/contrastive_pretrain/train_discriminator_asyn.sh