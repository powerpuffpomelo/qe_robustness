#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.infl_knn \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc.yaml" \
    --faiss_index_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/faiss_index/" \
    --recursion_depth 5000 \
    --test_id 2 \
    --test_token 1 \
    --scale 150.0 \
    --use_gpu 

# bash scripts/attribution_enzh/infl_knn_2.sh