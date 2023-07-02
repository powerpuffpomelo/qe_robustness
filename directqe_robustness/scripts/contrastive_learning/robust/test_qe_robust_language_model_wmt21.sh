#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/language_model_wmt21/discriminator_finetuned_256_wmt21_language_model/discriminator.best.ckpt.27600" \
    --source_path "/data1/yanym/data/robust_language_model_wmt21/test_limit3_10/robust_test.src.BPE" \
    --target_path "/data1/yanym/data/robust_language_model_wmt21/test_limit3_10/robust_test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --robust_idx_path "/data1/yanym/data/robust_language_model_wmt21/test_limit3_10/robust_test.idx.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/language_model_wmt21/discriminator_finetuned_256_wmt21_language_model/test_result_27600/robust_test_word_alignment.tag_idxonly.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/language_model_wmt21/discriminator_finetuned_256_wmt21_language_model/test_result_27600
GOLD_PREFIX=/data1/yanym/data/robust_language_model_wmt21/test_limit3_10
python estimate_word.py $GOLD_PREFIX/robust_test.tag_idxonly.BPE $PRE_PREFIX/robust_test_word_alignment.tag_idxonly.BPE

# bash scripts/contrastive_learning/robust/test_qe_robust_language_model_wmt21.sh