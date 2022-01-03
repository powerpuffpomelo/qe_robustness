#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/train.en_100w.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/train.zh_100w" \
    --config_path "./configs/analysis/qe_enzh_knn.yaml" \
    --batch_size 1 \
    --saveto_hter "/home/yanym/save/enzh/knn_predict/parallel_pred.hter" \
    --saveto_tags "/home/yanym/save/enzh/knn_predict/parallel_pred.tag" \
    --use_gpu

# bash scripts/train_directqe_enzh_infl/test_qe.sh