#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_mlm_wwm \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/contrastive/pretrain_generator_zhen.yaml" \
    --log_path "/home/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generator_wwm" \
    --saveto "/home/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generator_wwm/" \
    --use_gpu


# bash scripts/contrastive_learning/contrastive_pretrain/train_generator_mlm_zhen_wwm.sh