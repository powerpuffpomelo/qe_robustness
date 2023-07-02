#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=6
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_token_mask \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/robust/qe_enzh_aug_wmt21.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/word_alignment_wmt21/discriminator_finetuned_256_wmt21_word_alignment_data_aug" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/word_alignment_wmt21/discriminator_finetuned_256_wmt21_word_alignment_data_aug" \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_word_alignment_data_aug.sh 1234