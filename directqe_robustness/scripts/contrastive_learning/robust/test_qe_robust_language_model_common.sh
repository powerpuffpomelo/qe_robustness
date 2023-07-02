#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200/discriminator.best.ckpt.18000" \
    --source_path "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/robust_test.src.BPE" \
    --target_path "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/robust_test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --robust_idx_path "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/robust_test.idx_common.BPE" \
    --saveto_tags "/data1/yanym/save/temp/robust_test_word2vec_limit_3_10.tag_common.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/temp
GOLD_PREFIX=/data1/yanym/data/robust_qe_test_word2vec_limit_3_10
python estimate_word.py $GOLD_PREFIX/robust_test.tag_common.BPE $PRE_PREFIX/robust_test_word2vec_limit_3_10.tag_common.BPE

# bash scripts/contrastive_learning/robust/test_qe_robust_language_model_common.sh