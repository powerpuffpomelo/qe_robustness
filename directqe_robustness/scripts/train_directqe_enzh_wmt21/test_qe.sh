#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21/discriminator.best.ckpt.14400" \
    --source_path "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh_wmt21.yaml" \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21/test_result_ckpt14400/test.hter" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21/test_result_ckpt14400/test.tag.BPE" \
    --use_gpu

# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_wmt21/test_result_ckpt14400
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test
python estimate_word.py $GOLD_PREFIX/test.mt_tag.BPE $PRE_PREFIX/test.tag.BPE
python estimate_sentence.py $PRE_PREFIX/test.hter $GOLD_PREFIX/test.hter

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_enzh_wmt21/test_qe.sh