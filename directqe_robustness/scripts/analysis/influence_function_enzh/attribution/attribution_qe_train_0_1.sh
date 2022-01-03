#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe_train \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh/" \
    --recursion_depth 5000 \
    --test_id 0 \
    --test_token 1 \
    --scale 25.0 \
    --use_gpu 

# bash scripts/attribution/attribution_qe_train_0_1.sh