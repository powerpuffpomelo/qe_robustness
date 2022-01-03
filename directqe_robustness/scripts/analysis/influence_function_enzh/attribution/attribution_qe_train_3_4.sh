#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe_train \
    --pretrain_path "./save_qe/discriminator.best.ende.final" \
    --config_path "./configs/qe_ende_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence/directqe_ende/" \
    --recursion_depth 5000 \
    --test_id 3 \
    --test_token 4 \
    --scale 25.0 \
    --use_gpu 

# bash scripts/attribution/attribution_qe_train_3_4.sh