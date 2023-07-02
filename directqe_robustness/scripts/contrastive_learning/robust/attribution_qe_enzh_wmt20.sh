#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/discriminator_finetuned_256_robust_aug_v1_mask2_plus/discriminator.best.ckpt.22800" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/analysis/gradient_analysis/qe_enzh.yaml" \
    --plot_path_prefix "/data1/yanym/save/temp_gradient/" \
    --align_bpe_path_mt2src "/data1/yanym/data/test_align/test.mt-src.alignments.BPE" \
    --gradients_type "g" \
    --test_id 0 \
    --ig_steps 100 \
    --batch_size 20 \
    --use_gpu 

# bash scripts/contrastive_learning/robust/attribution_qe_enzh_wmt20.sh