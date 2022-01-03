#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe \
    --pretrain_path "/home/yanym/save/directqe_enzh_bpe/discriminator_finetuned_contrastive_256_aug_bi_batch2_freq30_lambda20102/discriminator.best.ckpt.4320" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/analysis/gradient_analysis/qe_enzh.yaml" \
    --plot_path_prefix "/home/yanym/save/directqe_enzh_bpe/gradient_contrastive_20102/" \
    --align_bpe_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/test/test.src-mt.alignments.BPE" \
    --gradients_type "g" \
    --test_id 0 \
    --ig_steps 100 \
    --batch_size 20 \
    --use_gpu 

# bash scripts/contrastive_learning/attribution_qe_enzh_contrastive.sh