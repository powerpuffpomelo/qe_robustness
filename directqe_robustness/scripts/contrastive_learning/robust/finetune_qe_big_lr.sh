#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/robust/qe_enzh_big_lr.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/proceed_cautiously/discriminator_finetuned_256_big_lr_0.00001" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/proceed_cautiously/discriminator_finetuned_256_big_lr_0.00001" \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_big_lr.sh 1234