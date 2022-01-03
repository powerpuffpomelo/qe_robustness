#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/yanym/save/directqe_enzh/discriminator_finetuned_contrastive_256_aug_bi_batch2_lambda1/discriminator.best.ckpt.150" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/yanym/save/directqe_enzh/discriminator_finetuned_contrastive_256_aug_bi_batch2_lambda1/test_result_150/train.hter" \
    --saveto_tags "/home/yanym/save/directqe_enzh/discriminator_finetuned_contrastive_256_aug_bi_batch2_lambda1/test_result_150/train.tag" \
    --use_gpu


# test result
PRE_PREFIX=/home/yanym/save/directqe_enzh/discriminator_finetuned_contrastive_256_aug_bi_batch2_lambda1/test_result_150
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/train.tag $PRE_PREFIX/train.tag
python estimate_sentence.py $PRE_PREFIX/train.hter $GOLD_PREFIX/train.hter

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/contrastive_learning/test_qe_train.sh