#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe_train \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_token_sample7000_scale500_test/" \
    --recursion_depth 7000 \
    --scale 500.0 \
    --saliency_level 'token' \
    --test_id 0 \
    --test_token 3 \
    --use_gpu 

# bash scripts/attribution_enzh/attribution_qe_train_token_0_3.sh