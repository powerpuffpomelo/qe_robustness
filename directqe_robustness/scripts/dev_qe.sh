#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "./save_qe/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/qe/data/wmt19_ende/qe_data_bpe/dev.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/qe/data/wmt19_ende/qe_data_bpe/dev.mt.BPE" \
    --config_path "./configs/qe_ende.yaml" \
    --batch_size 20 \
    --saveto_hter "./result/dev.hter" \
    --saveto_tags "./result/dev.tag" \
    --use_gpu
