#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "./save_discriminator_enzh/discriminator.best.final" \
    --config_path "./configs/qe_enzh.yaml" \
    --log_path "./log_qe" \
    --saveto "./save_finetune_enzh" \
    --use_gpu


# sh scripts/finetune_qe.sh 1234 > logs/finetune_enzh.log