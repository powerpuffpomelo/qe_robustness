#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "./save_qe/discriminator.best.ende.final" \
    --source_path "/home/user_data_182b/yanym/qe/data/wmt19_ende/create_one_fault/create.src.BPE" \
    --target_path "/home/user_data_182b/yanym/qe/data/wmt19_ende/create_one_fault/create.mt.BPE" \
    --config_path "./configs/qe_ende.yaml" \
    --batch_size 20 \
    --saveto_hter "./result/test_create.hter" \
    --saveto_tags "./result/test_create.tag" \
    --use_gpu

# scripts/test_qe_create.sh > mello_log/test.log