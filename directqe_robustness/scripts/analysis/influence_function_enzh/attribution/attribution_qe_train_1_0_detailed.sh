#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe_train_detailed \
    --pretrain_path "./save_qe/discriminator.best.ende.final" \
    --config_path "./configs/qe_ende_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence/directqe_ende/" \
    --recursion_depth 1000 \
    --test_id 1 \
    --test_token 0 \
    --num_first 20 \
    --use_gpu 

# bash scripts/attribution/attribution_qe_train_1_0_detailed.sh