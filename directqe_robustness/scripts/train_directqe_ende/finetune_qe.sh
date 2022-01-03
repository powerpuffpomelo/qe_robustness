#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/ende/discriminator_pretrained_256/discriminator.best.final" \
    --config_path "./configs/train_directqe_ende/qe_ende.yaml" \
    --log_path "/home/yanym/save/ende/discriminator_finetuned_256" \
    --saveto "/home/yanym/save/ende/discriminator_finetuned_256" \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_ende/finetune_qe.sh 1234