#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200/discriminator.best.ckpt.18000" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/train_directqe_enzh/qe_enzh.yaml" \
    --plot_path_prefix "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/gradient_plot_temp/" \
    --align_bpe_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/test/test.mt-src.alignments.BPE" \
    --gradients_type "g" \
    --test_id 0 \
    --ig_steps 100 \
    --batch_size 20 \
    --use_gpu 

# bash scripts/contrastive_learning/contrastive_pretrain/attribution_qe_enzh.sh