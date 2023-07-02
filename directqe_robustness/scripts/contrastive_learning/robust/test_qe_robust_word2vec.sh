#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_b_mask2_plus_word2vec_double/discriminator.best.ckpt.66000" \
    --source_path "/data1/yanym/data/robust_qe_test_word2vec/robust_test.src.BPE" \
    --target_path "/data1/yanym/data/robust_qe_test_word2vec/robust_test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --robust_idx_path "/data1/yanym/data/robust_qe_test_word2vec/robust_test.idx.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_b_mask2_plus_word2vec_double/test_result/robust_test_word2vec.tag_idxonly.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_b_mask2_plus_word2vec_double/test_result
GOLD_PREFIX=/data1/yanym/data/robust_qe_test_word2vec
python estimate_word.py $GOLD_PREFIX/robust_test.tag_idxonly.BPE $PRE_PREFIX/robust_test_word2vec.tag_idxonly.BPE

# bash scripts/contrastive_learning/robust/test_qe_robust_word2vec.sh