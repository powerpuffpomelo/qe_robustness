#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.new_finetune \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_pretrained_256/discriminator.best.final" \
    --config_path "./configs/train_directqe_enzh/new_finetune.yaml" \
    --log_path_prefix "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_sample5000_scale200/part_train_test_0/wo_n/" \
    --test_id 0 \
    --wo n \
    --use_gpu


# bash /home/data_ti4_c/yanym/qe/code/directqe_analysis/scripts/train_directqe_enzh/new_finetune/finetune_sample5000_scale200_test0_wo_n.sh 1234