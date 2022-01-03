#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/ende/discriminator_finetuned_256_with_parallel_cl_aug50/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version/test.mt.BPE" \
    --config_path "./configs/train_directqe_ende/qe_ende.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/yanym/save/ende/discriminator_finetuned_256_with_parallel_cl_aug50/test_result/test.hter" \
    --saveto_tags "/home/yanym/save/ende/discriminator_finetuned_256_with_parallel_cl_aug50/test_result/test.tag" \
    --use_gpu

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_ende/test_qe.sh