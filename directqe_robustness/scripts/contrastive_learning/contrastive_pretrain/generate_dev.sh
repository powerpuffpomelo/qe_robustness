#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.generate_dev_src \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/contrastive/generate_dev_enzh.yaml" \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generator_wwm/generator_mlm.best.final" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generate_dev" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generate_dev/" \
    --align_bpe_path "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/alignments/test.enzh_100w.alignments.BPE" \
    --split "test" \
    --use_gpu

# bash scripts/contrastive_learning/contrastive_pretrain/generate_dev.sh