#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_exp_v3_lambda0.5_add_wordemb/discriminator.best.ckpt.13200" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --align_bpe_path_mt2src_test "/data1/yanym/data/test_align_enzh/test.mt-src.alignments.BPE" \
    --align_ratio 0.5 \
    --add_align \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_exp_v3_lambda0.5_add_wordemb/test_result_13200/test.hter" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_exp_v3_lambda0.5_add_wordemb/test_result_13200/test.tag.BPE" \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_exp_v3_lambda0.5_add_wordemb/test_result_13200
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/test.tag.BPE $PRE_PREFIX/test.tag.BPE
python estimate_sentence.py $PRE_PREFIX/test.hter $GOLD_PREFIX/test.hter

# bash scripts/contrastive_learning/robust/test_qe_align_attn_add.sh