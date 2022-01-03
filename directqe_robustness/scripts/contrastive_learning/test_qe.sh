#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/yanym/save/directqe_enzh_bpe/discriminator_finetuned_contrastive_256_aug_bi_batch2_freq30_lambda20102/discriminator.best.ckpt.4800" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/yanym/save/directqe_enzh_bpe/discriminator_finetuned_contrastive_256_aug_bi_batch2_freq30_lambda20102/test_result_4800/test.hter" \
    --saveto_tags "/home/yanym/save/directqe_enzh_bpe/discriminator_finetuned_contrastive_256_aug_bi_batch2_freq30_lambda20102/test_result_4800/test.tag.BPE" \
    --use_gpu


# test result
PRE_PREFIX=/home/yanym/save/directqe_enzh_bpe/discriminator_finetuned_contrastive_256_aug_bi_batch2_freq30_lambda20102/test_result_4800
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/test.tag.BPE $PRE_PREFIX/test.tag.BPE
python estimate_sentence.py $PRE_PREFIX/test.hter $GOLD_PREFIX/test.hter

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/contrastive_learning/test_qe.sh