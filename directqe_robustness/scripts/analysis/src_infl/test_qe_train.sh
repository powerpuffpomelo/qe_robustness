#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/discriminator.best.final" \
    --source_path "/home/yanym/save/enzh/src_infl/order/train.src.3o_sample/train.src.3o_12138.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/yanym/save/enzh/src_infl/order/train_result_3o_sample/train.hter" \
    --saveto_tags "/home/yanym/save/enzh/src_infl/order/train_result_3o_sample/train.tag_12138" \
    --use_gpu

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/analysis/src_infl/test_qe_train.sh