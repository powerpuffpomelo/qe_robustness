#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.generate_dev_src_robust \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/contrastive/generate_dev_ende_robust.yaml" \
    --pretrain_path "/home/data_ti4_c/yanym/save/ende/generator/generator_mlm.best.final" \
    --log_path "/data1/yanym/data/robust_qe_test_ende/generate_dev" \
    --saveto "/data1/yanym/data/robust_qe_test_ende/generate_dev/" \
    --align_bpe_path_mt2src "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version_transquest/qe_test/test.mt-src.alignments.BPE" \
    --split "test" \
    --use_gpu

# bash scripts/contrastive_learning/robust/generate_robust_testset_ende.sh