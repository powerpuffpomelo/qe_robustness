#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe_train \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_both_toy10_scale100/" \
    --recursion_depth 10 \
    --scale 100.0 \
    --test_id 4 \
    --saliency_level "both" \
    --use_gpu 

# bash scripts/attribution_enzh/attribution_qe_train_both_0.sh