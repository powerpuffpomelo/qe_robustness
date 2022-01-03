#!/usr/bin/env bash
# test写入文件，每次都要打开文件太麻烦，这里搞一个直接显示预测结果标签的

export CUDA_VISIBLE_DEVICES=3

# echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_show \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --show_id 0 

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/analysis/gradient_analysis/test_qe_show.sh