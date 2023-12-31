#!/usr/bin/env bash
# 只关注训练样本数量多的token
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_only_many \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200/discriminator.best.ckpt.18000" \
    --config_path "./configs/analysis/training_data_analysis/qe_enzh_many.yaml" \
    --log_path "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many_100" \
    --saveto "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many_100" \
    --train_num_threshold 100 \
    --use_gpu


# bash scripts/analysis/training_data_analysis/finetune_qe_only_many.sh 1234