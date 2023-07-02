#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/language_model/discriminator_finetuned_256_language_model_add_feature_n2r/discriminator.best.ckpt.58800" \
    --source_path "/data1/yanym/data/robust_language_model/robust_qe_test_word2vec_limit_3_10/robust_test.src.BPE" \
    --target_path "/data1/yanym/data/robust_language_model/robust_qe_test_word2vec_limit_3_10/robust_test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --add_feature \
    --align_bpe_path_mt2src_test "/data1/yanym/data/robust_language_model/robust_qe_test_word2vec_limit_3_10/align/robust_test.mt-src.alignments.BPE.json" \
    --robust_idx_path "/data1/yanym/data/robust_language_model/robust_qe_test_word2vec_limit_3_10/robust_test.idx.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/language_model/discriminator_finetuned_256_language_model_add_feature_n2r/test_result_58800/robust_test_word2vec_limit_3_10.tag_idxonly.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/language_model/discriminator_finetuned_256_language_model_add_feature_n2r/test_result_58800
GOLD_PREFIX=/data1/yanym/data/robust_language_model/robust_qe_test_word2vec_limit_3_10
python estimate_word.py $GOLD_PREFIX/robust_test.tag_idxonly.BPE $PRE_PREFIX/robust_test_word2vec_limit_3_10.tag_idxonly.BPE

# bash scripts/contrastive_learning/robust/test_qe_robust_language_model.sh