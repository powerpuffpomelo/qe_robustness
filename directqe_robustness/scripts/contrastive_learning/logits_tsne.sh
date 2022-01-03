#!/usr/bin/env bash
# 根据需要，画logits图像

export CUDA_VISIBLE_DEVICES=2

# echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.logits_tsne \
    --pretrain_path_1 "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/discriminator.best.final" \
    --pretrain_path_2 "/home/yanym/save/directqe_enzh/discriminator_finetuned_contrastive_batch2_lambda1_freq30/discriminator.best.ckpt.6330" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --use_gpu

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/contrastive_learning/logits_tsne.sh