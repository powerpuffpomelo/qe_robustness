#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/discriminator_finetuned_256_align_attn_v2_lambda_0.5/discriminator.best.ckpt.13200" \
    --source_path "/data1/yanym/data/robust_qe_test/generate_dev/robust_test.src.BPE" \
    --target_path "/data1/yanym/data/robust_qe_test/generate_dev/robust_test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --robust_idx_path "/data1/yanym/data/robust_qe_test/generate_dev/robust_test.idx.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/discriminator_finetuned_256_align_attn_v2_lambda_0.5/test_result/robust_test.tag_idxonly.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/discriminator_finetuned_256_align_attn_v2_lambda_0.5/test_result
GOLD_PREFIX=/data1/yanym/data/robust_qe_test/generate_dev
python estimate_word.py $GOLD_PREFIX/robust_test.tag_idxonly.BPE $PRE_PREFIX/robust_test.tag_idxonly.BPE

# bash scripts/contrastive_learning/robust/test_qe_robust.sh