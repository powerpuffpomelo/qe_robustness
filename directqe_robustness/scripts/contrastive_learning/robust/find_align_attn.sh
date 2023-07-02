#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.find_align_attn \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/robust/qe_enzh.yaml" \
    --align_bpe_path_mt2src_train "/home/yanym/data/test_align/train.mt-src.alignments.BPE" \
    --log_path "/home/yanym/save/directqe_enzh_bpe_robust/find_align_attn_temp" \
    --saveto "/home/yanym/save/directqe_enzh_bpe_robust/find_align_attn_temp" \
    --use_gpu


# bash scripts/contrastive_learning/robust/find_align_attn.sh