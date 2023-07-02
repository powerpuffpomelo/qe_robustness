#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_a_all/discriminator.best.ckpt.25200" \
    --source_path "/data1/yanym/data/robust_qe_train_v1/robust_train.src.BPE" \
    --target_path "/data1/yanym/data/robust_qe_train_v1/robust_train.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_a_all/test_result_25200/train_real.hter" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_a_all/test_result_25200/train_real.tag.BPE" \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_a_all/test_result_25200
GOLD_PREFIX=/data1/yanym/data/robust_qe_train_v1
python estimate_word.py $GOLD_PREFIX/robust_train.tag.BPE $PRE_PREFIX/train_real.tag.BPE
python estimate_sentence.py $GOLD_PREFIX/robust_train.hter $PRE_PREFIX/train_real.hter

# bash scripts/contrastive_learning/robust/test_qe_train_real_a.sh