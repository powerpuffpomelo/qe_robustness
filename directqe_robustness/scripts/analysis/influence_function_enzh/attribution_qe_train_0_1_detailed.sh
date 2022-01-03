#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe_train_detailed \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/new_finetune/test2_wo_p/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh/" \
    --recursion_depth 1000 \
    --test_id 0 \
    --test_token 1 \
    --num_first 20 \
    --use_gpu 

# bash scripts/attribution_enzh/attribution_qe_train_0_1_detailed.sh