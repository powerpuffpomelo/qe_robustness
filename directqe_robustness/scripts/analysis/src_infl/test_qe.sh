#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/yanym/save/directqe_enzh/discriminator_finetuned_contrastive_batch2_lambda1_freq30/discriminator.best.ckpt.6330" \
    --source_path "/home/yanym/save/enzh/src_infl/order/test.src.3o_token/test.src.3o_12138.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/yanym/save/enzh/src_infl/order/test_result_3o_token_contrastive/test.hter" \
    --saveto_tags "/home/yanym/save/enzh/src_infl/order/test_result_3o_token_contrastive/test.tag_12138" \
    --use_gpu

# test result
PRE_PREFIX=/home/yanym/save/enzh/src_infl/order/test_result_3o_token_contrastive
#GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
GOLD_PREFIX=/home/yanym/save/directqe_enzh/discriminator_finetuned_contrastive_batch2_lambda1_freq30/test_result_6330
python estimate_word.py $GOLD_PREFIX/test.tag $PRE_PREFIX/test.tag_12138

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/analysis/src_infl/test_qe.sh