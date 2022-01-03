#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator_contrastive \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generator_wwm/generator_mlm.best.final" \
    --pretrain_discriminator_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/contrastive/pretrain_discriminator_enzh_contrastive.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_pretrained_256_contrastive_dynamic_from_normal_101" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_pretrained_256_contrastive_dynamic_from_normal_101/" \
    --align_bpe_path "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/parallel_plus_pseudo_from_src/alignments/parallel_train.srcmt.alignments.BPE" \
    --use_gpu

# bash scripts/contrastive_learning/contrastive_pretrain/train_discriminator_contrastive.sh