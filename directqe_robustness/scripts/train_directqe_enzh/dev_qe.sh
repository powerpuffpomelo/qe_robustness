#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/yanym/save/enzh/discriminator_finetuned_512_log_v4_worse_pseudo_mt_sent_temp/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/dev.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/dev.mt" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/yanym/save/enzh/discriminator_finetuned_512_log_v4_worse_pseudo_mt_sent_temp/test_result/dev.hter" \
    --saveto_tags "/home/yanym/save/enzh/discriminator_finetuned_512_log_v4_worse_pseudo_mt_sent_temp/test_result/dev.tag" \
    --use_gpu

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_enzh/dev_qe.sh
