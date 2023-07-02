#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_robust_aug_anti \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200/discriminator.best.ckpt.18000" \
    --config_path "./configs/robust/qe_enzh.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/discriminator_finetuned_256_robust_aug_anti_lambda0.1_pad" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/discriminator_finetuned_256_robust_aug_anti_lambda0.1_pad" \
    --aug_lambda 0.1 \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_robust_aug_anti.sh 1234