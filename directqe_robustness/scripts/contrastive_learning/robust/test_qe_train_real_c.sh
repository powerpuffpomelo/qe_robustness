#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_c_mask1/discriminator.best.ckpt.32400" \
    --source_path "/data1/yanym/data/robust_qe_train_v1/robust_train.src.BPE" \
    --target_path "/data1/yanym/data/robust_qe_train_v1/robust_train.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --robust_idx_path "/data1/yanym/data/robust_qe_train_v1/robust_train.idx.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_c_mask1/test_result_32400/train_real.tag.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_imp_v1_c_mask1/test_result_32400
GOLD_PREFIX=/data1/yanym/data/robust_qe_train_v1
python estimate_word.py $GOLD_PREFIX/robust_train.tag.only_idx.BPE $PRE_PREFIX/train_real.tag.BPE

# bash scripts/contrastive_learning/robust/test_qe_train_real_c.sh