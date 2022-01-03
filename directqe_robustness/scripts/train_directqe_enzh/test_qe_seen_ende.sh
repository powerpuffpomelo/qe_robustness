#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/ende/discriminator_after_finetune/discriminator.best.final4_1" \
    --source_path "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/seen_test/unseen_test.src.BPE" \
    --target_path "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/seen_test/unseen_test.mt" \
    --config_path "./configs/qe_ende.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/seen_test/unseen_test_pre.hter" \
    --saveto_tags "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/seen_test/unseen_test_pre.tag" \
    --use_gpu

# bash /home/data_ti4_c/yanym/qe/code/directqe_analysis/scripts/train_directqe_enzh/test_qe_seen_ende.sh