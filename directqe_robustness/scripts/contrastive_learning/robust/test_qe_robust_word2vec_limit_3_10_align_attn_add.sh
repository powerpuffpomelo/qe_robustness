#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_exp_v3_lambda0.5_add_wordemb/discriminator.best.ckpt.13200" \
    --source_path "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/robust_test.src.BPE" \
    --target_path "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/robust_test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --align_bpe_path_mt2src_test "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/align/robust_test.mt-src.alignments.BPE" \
    --align_ratio 0.5 \
    --add_align \
    --robust_idx_path "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/robust_test.idx.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_exp_v3_lambda0.5_add_wordemb/test_result_13200/robust_test.tag_idxonly.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/imp_1finetune/discriminator_finetuned_256_exp_v3_lambda0.5_add_wordemb/test_result_13200
GOLD_PREFIX=/data1/yanym/data/robust_qe_test_word2vec_limit_3_10
python estimate_word.py $GOLD_PREFIX/robust_test.tag_idxonly.BPE $PRE_PREFIX/robust_test.tag_idxonly.BPE

# bash scripts/contrastive_learning/robust/test_qe_robust_word2vec_limit_3_10_align_attn_add.sh