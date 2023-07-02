#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory_wmt21/discriminator_finetuned_256_wmt21_train_memory_data_aug/discriminator.best.ckpt.8400" \
    --source_path "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --robust_idx_path "/data1/yanym/data/robust_train_memory_wmt21/test.niche_idx.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory_wmt21/discriminator_finetuned_256_wmt21_train_memory_data_aug/test_result_8400/robust_test_train_memory.tag_idxonly.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/train_memory_wmt21/discriminator_finetuned_256_wmt21_train_memory_data_aug/test_result_8400
GOLD_PREFIX=/data1/yanym/data/robust_train_memory_wmt21
python estimate_word.py $GOLD_PREFIX/robust_test.tag_idxonly.BPE $PRE_PREFIX/robust_test_train_memory.tag_idxonly.BPE

# bash scripts/contrastive_learning/robust/test_qe_robust_train_memory_wmt21.sh