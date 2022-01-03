#!/usr/bin/env bash
# 是一个一边预训练一边记录结果loss的实验

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator_no_need_2_generate_log \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_lm_forward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_forward.best.final" \
    --pretrain_generator_lm_backward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_backward.best.final" \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/enzh/generator/generator_mlm.best.final" \
    --config_path "./configs/train_directqe_enzh/pretrain_discriminator_enzh_log_qe_pseudo.yaml" \
    --log_path "/home/yanym/save/enzh/discriminator_pretrained_256_log_v5_qe_pseudo_part_v4" \
    --saveto "/home/yanym/save/enzh/discriminator_pretrained_256_log_v5_qe_pseudo_part_v4/" \
    --use_gpu

# bash scripts/train_directqe_enzh/train_discriminator_log.sh