#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.knn_analysis \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_knn.yaml" \
    --faiss_index_prefix "/home/yanym/save/enzh/faiss_index/" \
    --knn_log_prefix "/home/yanym/save/enzh/knn_log/" \
    --test_id 2 \
    --test_token 1 \
    --use_gpu 

# bash scripts/train_directqe_enzh_infl/knn/knn_analysis_2.sh