#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_align_attn_v3 \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/robust/qe_enzh.yaml" \
    --align_bpe_path_mt2src_train "/data1/yanym/data/test_align_enzh/train.mt-src.alignments.BPE" \
    --align_bpe_path_mt2src_dev "/data1/yanym/data/test_align_enzh/dev.mt-src.alignments.BPE" \
    --attn_align_lambda 0.5 \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_exp_v3_lambda0.5_add_wordemb" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_exp_v3_lambda0.5_add_wordemb" \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_exp_v3_add_word_embedding.sh 1234