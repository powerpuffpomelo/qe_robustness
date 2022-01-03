#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe_train \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_only_qe_256_1layer/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc_1layer.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/infl_check/directqe_enzh_token_sample1000_scale100/" \
    --recursion_depth 1000 \
    --scale 100.0 \
    --saliency_level 'token' \
    --test_id 0 \
    --test_token 3 \
    --use_gpu 

# bash /home/data_ti4_c/yanym/qe/code/directqe_analysis/scripts/train_directqe_enzh_infl/infl_check/attribution_qe_train_token_0_3.sh