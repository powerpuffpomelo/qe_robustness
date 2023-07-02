#!/usr/bin/env bash
# test写入文件，每次都要打开文件太麻烦，这里搞一个直接显示预测结果标签的

export CUDA_VISIBLE_DEVICES=3

# echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_show_onecase \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/discriminator.best.ckpt.9600" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --src_sent "in 18@@ 89 , as much as 5@@ 0,000 cor@@ ds of t@@ an@@ bar@@ k were h@@ au@@ led out from the Li@@ t@@ tle S@@ ur Ri@@ ver and Big S@@ ur Ri@@ ver water@@ she@@ ds ." \
    --mt_sent "1889 年 ， 从小 苏尔 河@@ 和@@ 大 苏尔 河 流域 拖@@ 出 了 多达 5 万@@ 条 油轮 。" \

# bash scripts/contrastive_learning/test_qe_show_onecase.sh