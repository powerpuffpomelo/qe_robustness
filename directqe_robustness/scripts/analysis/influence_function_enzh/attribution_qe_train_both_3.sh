#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe_train \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_both_sample7k_scale10w/" \
    --recursion_depth 7000 \
    --scale 100000.0 \
    --test_id 3 \
    --saliency_level "both" \
    --use_gpu 

# bash scripts/attribution_enzh/attribution_qe_train_both_3.sh