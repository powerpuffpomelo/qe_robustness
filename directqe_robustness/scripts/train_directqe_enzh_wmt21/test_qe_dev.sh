#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21/discriminator.best.ckpt.18000" \
    --source_path "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_dev/dev.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_dev/dev.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh_wmt21.yaml" \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21/test_result_ckpt18000/dev.hter" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21/test_result_ckpt18000/dev.tag.BPE" \
    --use_gpu

# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21/test_result_ckpt18000
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_dev
python estimate_word.py $GOLD_PREFIX/dev.mt_tag.BPE $PRE_PREFIX/dev.tag.BPE
python estimate_sentence.py $PRE_PREFIX/dev.hter $GOLD_PREFIX/dev.hter

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_enzh_wmt21/test_qe_dev.sh