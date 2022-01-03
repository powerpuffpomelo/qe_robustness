#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_toy_test/discriminator_finetuned_256_naive_aug_v3/discriminator.best.ckpt.90" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/analysis/gradient_analysis/qe_enzh.yaml" \
    --plot_path_prefix "/data1/yanym/save/directqe_enzh_bpe_toy_test/discriminator_finetuned_256_naive_aug_v3/gradient/" \
    --align_bpe_path_mt2src "/data1/yanym/data/test_align/train.mt-src.alignments.BPE" \
    --gradients_type "g" \
    --test_id 0 \
    --ig_steps 100 \
    --batch_size 20 \
    --use_gpu 

# bash scripts/contrastive_learning/robust/toy_test/attribution_qe_enzh.sh