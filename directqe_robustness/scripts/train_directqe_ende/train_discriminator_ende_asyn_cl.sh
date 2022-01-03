#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.train_discriminator_no_need_2_generate_cl \
    --model_name "discriminator" \
    --reload \
    --pretrain_generator_lm_forward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_forward.best.final" \
    --pretrain_generator_lm_backward_path "/home/user_data55/cuiq/Direct_QE/save/generator_lm_backward.best.final" \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/ende/generator_temp/generator_mlm.best.final" \
    --config_path "./configs/train_directqe_ende/pretrain_discriminator_ende_asyn_cl.yaml" \
    --log_path "/home/yanym/save/ende/discriminator_pretrained_256_with_parallel_cl_aug10" \
    --saveto "/home/yanym/save/ende/discriminator_pretrained_256_with_parallel_cl_aug10/" \
    --cl_aug 10 \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_ende/train_discriminator_ende_asyn_cl.sh