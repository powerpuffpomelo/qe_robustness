#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_pretrained_256_log_v4_worse_pseudo_mt/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many/test_result_begin/test.hter" \
    --saveto_tags "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many/test_result_begin/test.tag" \
    --use_gpu

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/analysis/training_data_analysis/test_qe_begin.sh