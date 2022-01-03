#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/train_directqe_enzh/qe_enzh_wmt21.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21" \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_enzh_wmt21/finetune_qe.sh 1234