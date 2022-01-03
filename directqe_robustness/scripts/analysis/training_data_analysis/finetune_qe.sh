#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_pretrained_256_log_v4_worse_pseudo_mt/discriminator.best.final" \
    --config_path "./configs/analysis/training_data_analysis/qe_enzh_filter.yaml" \
    --log_path "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_wolf_bad" \
    --saveto "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_wolf_bad" \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/analysis/training_data_analysis/finetune_qe.sh 1234