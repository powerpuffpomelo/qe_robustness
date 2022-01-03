#!/usr/bin/env bash
# test写入文件，每次都要打开文件太麻烦，这里搞一个直接显示预测结果标签的

export CUDA_VISIBLE_DEVICES=3

# echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_show_onecase \
    --pretrain_path "/home/yanym/save/directqe_enzh/discriminator_finetuned_contrastive_256_bi_batch2_lambda1/discriminator.best.ckpt.1" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --src_sent "he sho@@ ves O@@ w@@ en into the p@@ it where Di@@ g@@ ger ri@@ ps out his son 's heart ." \
    --mt_sent "他 把 欧文 扔进 了 挖掘机 挖出 儿子 心脏 的 坑里 ." \

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/contrastive_learning/test_qe_show_onecase_cl.sh