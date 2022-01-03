#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.knn_analysis \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_ckpt/discriminator.best.ckpt.22800" \
    --config_path "./configs/analysis/knn_analysis/qe_enzh_knn.yaml" \
    --feature_prefix "/home/data_ti4_c/yanym/save/enzh/knn/faiss_index/" \
    --knn_log_prefix "/home/yanym/save/enzh/knn_log_token/" \
    --search_level "token" \
    --feature_model "directqe" \
    --test_id 1 \
    --test_token 8 \
    --use_gpu 

# bash scripts/analysis/knn_analysis/knn_analysis.sh