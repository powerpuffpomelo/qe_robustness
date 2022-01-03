#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_align_attn \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/robust/qe_enzh_toy.yaml" \
    --align_bpe_path_mt2src "/data1/yanym/data/test_align/train.mt-src.alignments.BPE" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_toy_test/discriminator_finetuned_256_naive_aug_temp" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_toy_test/discriminator_finetuned_256_naive_aug_temp" \
    --use_gpu


# bash scripts/contrastive_learning/robust/toy_test/finetune_qe_align_attn.sh 1234