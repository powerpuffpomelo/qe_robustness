#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=6
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_token_mask \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200/discriminator.best.ckpt.18000" \
    --config_path "./configs/robust/qe_enzh_aug.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/word_alignment/discriminator_finetuned_256_word_alignment_data_aug_2finetune" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/word_alignment/discriminator_finetuned_256_word_alignment_data_aug_2finetune" \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_imp_v1_mask.sh 1234