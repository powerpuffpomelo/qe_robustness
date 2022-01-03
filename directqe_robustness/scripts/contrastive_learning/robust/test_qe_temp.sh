#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/discriminator_finetuned_256_robust_aug_v1/discriminator.best.ckpt.27600" \
    --source_path "/data1/yanym/data/robust_qe_test/generate_dev/robust_test.src.BPE" \
    --target_path "/data1/yanym/data/robust_qe_test/generate_dev/robust_test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/data/robust_qe_test_analysis/qe_robust_test_result/v1_test.hter" \
    --saveto_tags "/data1/yanym/data/robust_qe_test_analysis/qe_robust_test_result/v1.tag.BPE" \
    --use_gpu


# bash scripts/contrastive_learning/robust/test_qe_temp.sh