#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_pretrained_512/discriminator.best.final" \
    --config_path "./configs/train_directqe_enzh/qe_enzh_1layer.yaml" \
    --log_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_only_qe_256_1layer" \
    --saveto "/home/user_data_182b/yanym/qe/save/enzh/discriminator_only_qe_256_1layer" \
    --use_gpu


# bash /home/data_ti4_c/yanym/qe/code/directqe_analysis/scripts/train_directqe_enzh_infl/train_qe_1layer.sh 1234