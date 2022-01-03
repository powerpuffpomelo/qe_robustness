#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_robust_aug_dynamic \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_generator_mlm_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generator_wwm/generator_mlm.best.final" \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200/discriminator.best.ckpt.18000" \
    --config_path "./configs/robust/qe_enzh_toy.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_toy_test/discriminator_finetuned_256_naive_aug_v3" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_toy_test/discriminator_finetuned_256_naive_aug_v3" \
    --align_bpe_path_mt2src "/data1/yanym/data/test_align/train.mt-src.alignments.BPE" \
    --use_gpu


# bash scripts/contrastive_learning/robust/toy_test/finetune_qe_robust_aug_dynamic_toy.sh 1234