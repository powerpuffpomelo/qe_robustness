#!/usr/bin/env bash
# test写入文件，每次都要打开文件太麻烦，这里搞一个直接显示预测结果标签的

export CUDA_VISIBLE_DEVICES=3

# echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_show_onecase \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/discriminator.best.final" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --src_sent "he sho@@ ves O@@ w@@ en into the p@@ it where Di@@ g@@ ger ri@@ ps out his son 's heart ." \
    --mt_sent "最后 的 征服者 骑着 他 的 剑 继续前进 ." \

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/contrastive_learning/test_qe_show_onecase.sh