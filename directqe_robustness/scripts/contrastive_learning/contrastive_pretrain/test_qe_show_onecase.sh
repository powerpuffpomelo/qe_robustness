#!/usr/bin/env bash
# test写入文件，每次都要打开文件太麻烦，这里搞一个直接显示预测结果标签的

export CUDA_VISIBLE_DEVICES=4

# echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_show_onecase \
    --pretrain_path "/data1/yanym/data/toy_contrastive/discriminator_normal_small/discriminator.best.ckpt.52" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --src_sent "27 . Appro@@ pri@@ ate legal frame@@ works supported by robust land policies were required , together with proper tools for their implementation ." \
    --mt_sent "弗朗@@ 兹 · 弗@@ 山" \
    --use_gpu

# bash scripts/contrastive_learning/contrastive_pretrain/test_qe_show_onecase.sh