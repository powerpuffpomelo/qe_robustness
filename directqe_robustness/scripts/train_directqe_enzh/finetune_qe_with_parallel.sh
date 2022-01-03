#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_pretrained_256_with_parallel/discriminator.best.final" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --log_path "/home/yanym/save/enzh/discriminator_finetuned_256_with_parallel" \
    --saveto "/home/yanym/save/enzh/discriminator_finetuned_256_with_parallel" \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_enzh/finetune_qe_with_parallel.sh 1234