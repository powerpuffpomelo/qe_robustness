#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_with_parallel_cl_aug10/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt" \
    --config_path "./configs/analysis/qe_enzh_knn_cl.yaml" \
    --batch_size 1 \
    --saveto_hter "/home/yanym/save/enzh/cl_aug10/test_result_qe_train/qe_train_pred.hter" \
    --saveto_tags "/home/yanym/save/enzh/cl_aug10/test_result_qe_train/qe_train_pred.tag" \
    --use_gpu

# bash scripts/analysis/knn_analysis/pred_qe_train.sh