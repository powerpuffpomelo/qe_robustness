#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator_contrastive \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generator_wwm/generator_mlm.best.final" \
    --config_path "./configs/contrastive/pretrain_discriminator_enzh_contrastive_temp.yaml" \
    --log_path "/data1/yanym/data/toy_contrastive/discriminator_contrastive_dynamic_debug_dynamic_1000" \
    --saveto "/data1/yanym/data/toy_contrastive/discriminator_contrastive_dynamic_debug_dynamic_1000/" \
    --align_bpe_path "/data1/yanym/data/toy_contrastive/data_1000/toy_parallel_train.srcmt.alignments.BPE" \
    --use_gpu

# bash scripts/contrastive_learning/contrastive_pretrain/train_discriminator_contrastive_temp.sh