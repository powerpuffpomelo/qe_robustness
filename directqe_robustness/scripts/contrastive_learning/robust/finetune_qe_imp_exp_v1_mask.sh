#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_token_mask_align_attn_v1 \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/robust/qe_enzh_aug.yaml" \
    --align_bpe_path_mt2src_train "/data1/yanym/data/robust_qe_train_word2vec_double/align/robust_train.mt-src.alignments.BPE" \
    --align_bpe_path_mt2src_dev "/data1/yanym/data/test_align_enzh/dev.mt-src.alignments.BPE" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_b_mask2_plus_word2vec_double_exp_v1_lambda0.5" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_b_mask2_plus_word2vec_double_exp_v1_lambda0.5" \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_imp_exp_v1_mask.sh 1234