#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_pretrained_256_log_v4_worse_pseudo_mt/discriminator.best.final" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/analysis/contrastive_learning/qe_enzh_aug.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh/discriminator_finetuned_256_aug2_real_switch" \
    --saveto "/data1/yanym/save/directqe_enzh/discriminator_finetuned_256_aug2_real_switch" \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/contrastive_learning/finetune_qe_aug.sh 1234