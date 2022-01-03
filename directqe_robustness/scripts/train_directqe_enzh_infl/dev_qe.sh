#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt" \
    --config_path "./configs/analysis/qe_enzh_knn.yaml" \
    --batch_size 800 \
    --saveto_hter "/home/yanym/save/enzh/model_predict/test.hter_pred" \
    --saveto_tags "/home/yanym/save/enzh/model_predict/pseudo_pred.tag" \
    --use_gpu

# bash scripts/train_directqe_enzh_infl/dev_qe.sh