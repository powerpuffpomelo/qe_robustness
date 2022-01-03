#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/ende/discriminator_finetuned_256_with_parallel_cl_aug50/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version/dev.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version/dev.mt.BPE" \
    --config_path "./configs/train_directqe_ende/qe_ende.yaml" \
    --batch_size 20 \
    --saveto_hter "/home/yanym/save/ende/discriminator_finetuned_256_with_parallel_cl_aug50/test_result/dev.hter" \
    --saveto_tags "/home/yanym/save/ende/discriminator_finetuned_256_with_parallel_cl_aug50/test_result/dev.tag" \
    --use_gpu

# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_ende/dev_qe.sh