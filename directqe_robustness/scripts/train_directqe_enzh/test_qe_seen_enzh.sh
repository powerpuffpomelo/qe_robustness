#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --source_path "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/seen_test/unseen_test.src.BPE" \
    --target_path "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/seen_test/unseen_test.mt" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/seen_test/unseen_test_pre.hter" \
    --saveto_tags "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/seen_test/unseen_test_pre.tag" \
    --use_gpu

# bash /home/data_ti4_c/yanym/qe/code/directqe_analysis/scripts/train_directqe_enzh/test_qe_seen_enzh.sh