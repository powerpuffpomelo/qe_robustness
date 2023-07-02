#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug_every_sample/discriminator.best.ckpt.5" \
    --source_path "/data1/yanym/data/robust_train_memory/data_augmentation_train_update/robust_train.src.BPE" \
    --target_path "/data1/yanym/data/robust_train_memory/data_augmentation_train_update/robust_train.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --robust_idx_path "/data1/yanym/data/robust_train_memory/data_augmentation_train_update/robust_train.mt_idx_contrast.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug_every_sample/test_result_5/robust_test_train_memory.tag_idxonly_contrast.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
#PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug_every_sample/test_result_5
#GOLD_PREFIX=/data1/yanym/data/robust_train_memory
#python estimate_word.py $GOLD_PREFIX/test.tag_niche_idxonly.BPE $PRE_PREFIX/robust_test_train_memory.tag_idxonly.BPE

# bash scripts/contrastive_learning/robust/test_qe_robust_train_memory_temp.sh