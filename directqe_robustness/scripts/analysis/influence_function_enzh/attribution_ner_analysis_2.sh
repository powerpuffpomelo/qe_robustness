#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_ner_analysis \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --config_path "./configs/analysis/qe_enzh_infunc.yaml" \
    --influence_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_sample5000_scale200/" \
    --ner_path_prefix "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/NER/" \
    --ner_lang "src" \
    --test_id 3 \
    --test_token 1 \
    --use_gpu 

# bash scripts/attribution_enzh/attribution_ner_analysis_2.sh