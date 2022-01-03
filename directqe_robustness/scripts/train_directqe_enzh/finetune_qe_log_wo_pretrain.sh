#!/usr/bin/env bash
# finetune时记录loss等信息，以备画图
# 无预训练，直接微调
export CUDA_VISIBLE_DEVICES=3
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_log \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --config_path "./configs/train_directqe_enzh/qe_enzh_qe_pseudo_temp.yaml" \
    --log_path "/home/yanym/save/enzh/discriminator_finetuned_256_log_v5_qe_pseudo_part_wo_pretrain_pseudo" \
    --saveto "/home/yanym/save/enzh/discriminator_finetuned_256_log_v5_qe_pseudo_part_wo_pretrain_pseudo" \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_enzh/finetune_qe_log_wo_pretrain.sh 1234