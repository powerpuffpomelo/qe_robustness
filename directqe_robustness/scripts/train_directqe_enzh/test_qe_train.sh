#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many/test_result/train.hter" \
    --saveto_tags "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many/test_result/train.tag" \
    --use_gpu

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_enzh/test_qe_train.sh