#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe_train \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/infl_check/directqe_enzh_both_traintest_pretrain_sample10_scale1000/" \
    --recursion_depth 10 \
    --scale 1000.0 \
    --test_id 5 \
    --saliency_level "both" \
    --use_gpu 

# bash scripts/train_directqe_enzh_infl/infl_check/attribution_qe_train_both_1.sh