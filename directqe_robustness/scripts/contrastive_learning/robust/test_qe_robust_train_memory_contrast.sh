#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_robust \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug/discriminator.best.ckpt.186000" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --robust_idx_path "/data1/yanym/data/robust_train_memory/test.popular_idx.BPE" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug/test_result_186000/robust_test_train_memory.tag_popular_idxonly.BPE" \
    --batch_size 20 \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug/test_result_186000
GOLD_PREFIX=/data1/yanym/data/robust_train_memory
python estimate_word.py $GOLD_PREFIX/test.tag_popular_idxonly.BPE $PRE_PREFIX/robust_test_train_memory.tag_popular_idxonly.BPE

# bash scripts/contrastive_learning/robust/test_qe_robust_train_memory_contrast.sh