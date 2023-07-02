#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200/discriminator.best.ckpt.18000" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/save/temp/train.hter" \
    --saveto_tags "/data1/yanym/save/temp/train.tag.BPE" \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/temp
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/train.tag.BPE $PRE_PREFIX/train.tag.BPE
python estimate_sentence.py $PRE_PREFIX/train.hter $GOLD_PREFIX/train.hter

# bash scripts/contrastive_learning/robust/test_qe_train_temp.sh