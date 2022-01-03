#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_sample5000_scale200/part_train_test_2/wo_p/discriminator.best.final" \
    --source_path "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/dev.src.BPE" \
    --target_path "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/dev.mt" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_sample5000_scale200/part_train_test_2/wo_p/test_result/dev.hter" \
    --saveto_tags "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_sample5000_scale200/part_train_test_2/wo_p/test_result/dev.tag" \
    --use_gpu

# bash /home/data_ti4_c/yanym/qe/code/directqe_analysis/scripts/train_directqe_enzh/new_finetune/new_dev_qe.sh