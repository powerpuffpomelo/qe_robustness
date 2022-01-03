#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_finetuned_256_contrastive_check8_from_normal_from_12640/discriminator.best.ckpt.22800" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_finetuned_256_contrastive_check8_from_normal_from_12640/test_result_22800/train.hter" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_finetuned_256_contrastive_check8_from_normal_from_12640/test_result_22800/train.tag.BPE" \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_finetuned_256_contrastive_check8_from_normal_from_12640/test_result_22800
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/train.tag.BPE $PRE_PREFIX/train.tag.BPE
python estimate_sentence.py $PRE_PREFIX/train.hter $GOLD_PREFIX/train.hter

# bash scripts/contrastive_learning/contrastive_pretrain/test_qe_temp.sh