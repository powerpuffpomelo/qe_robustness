#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200_noshuffle" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200_noshuffle" \
    --use_gpu


# bash scripts/train_directqe_enzh/finetune_qe.sh 1234