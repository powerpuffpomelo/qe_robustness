#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_b_mask2_plus/discriminator.best.ckpt.55200" \
    --source_path "/data1/yanym/data/robust_qe_train_v1/robust_train.src.BPE" \
    --target_path "/data1/yanym/data/robust_qe_train_v1/robust_train.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --robust_idx_path "/data1/yanym/data/robust_qe_train_v1/robust_train.idx_plus.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_b_mask2_plus/test_result_55200/train_real.tag.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_b_mask2_plus/test_result_55200
GOLD_PREFIX=/data1/yanym/data/robust_qe_train_v1
python estimate_word.py $GOLD_PREFIX/robust_train.tag.only_idx_plus.BPE $PRE_PREFIX/train_real.tag.BPE

# bash scripts/contrastive_learning/robust/test_qe_train_real_b.sh