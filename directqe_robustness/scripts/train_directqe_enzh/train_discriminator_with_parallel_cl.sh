#!/usr/bin/env bash
# 加了对比学习的，平行语料和伪数据对比

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator_no_need_2_generate_cl \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_lm_forward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_forward.best.final" \
    --pretrain_generator_lm_backward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_backward.best.final" \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/enzh/generator/generator_mlm.best.final" \
    --config_path "./configs/train_directqe_enzh/pretrain_discriminator_enzh_with_parallel_cl.yaml" \
    --log_path "/home/yanym/save/enzh/discriminator_pretrained_256_with_parallel_cl_aug50" \
    --saveto "/home/yanym/save/enzh/discriminator_pretrained_256_with_parallel_cl_aug50/" \
    --cl_aug 50 \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_enzh/train_discriminator_with_parallel_cl.sh