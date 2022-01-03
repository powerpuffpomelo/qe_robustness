#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.infl_tools \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_sample5000_scale200/part_train_test_1/wo_p/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/occupy/" \
    --task "cal_test_loss" \
    --recursion_depth 5000 \
    --test_id 1 \
    --test_token 1 \
    --scale 200.0 \
    --use_gpu 

# bash scripts/attribution_enzh/infl_tools.sh