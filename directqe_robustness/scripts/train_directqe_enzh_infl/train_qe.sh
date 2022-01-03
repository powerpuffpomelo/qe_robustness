#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_pretrained_512/discriminator.best.final" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --log_path "/home/user_data_182b/yanym/qe/save/enzh/discriminator_only_qe_256" \
    --saveto "/home/user_data_182b/yanym/qe/save/enzh/discriminator_only_qe_256" \
    --use_gpu


# bash /home/data_ti4_c/yanym/qe/code/directqe_analysis/scripts/train_directqe_enzh_infl/train_qe.sh 1234