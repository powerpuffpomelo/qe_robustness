#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_knn_token \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --source_path "/home/yanym/save/enzh/knn/data_temp/test.src.BPE" \
    --target_path "/home/yanym/save/enzh/knn/data_temp/test.mt" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/analysis/knn_analysis/qe_enzh_knn.yaml" \
    --batch_size 20 \
    --k 10 \
    --temperature 0.1 \
    --lambda_knn 1 \
    --faiss_index_prefix "/home/data_ti4_c/yanym/save/enzh/knn/faiss_index/" \
    --saveto_hter "/home/yanym/save/enzh/knn_predict/test.hter" \
    --saveto_tags "/home/yanym/save/enzh/knn/test_qe_train_k10_t0.1_lambda1.tag" \
    --use_gpu

# bash scripts/knn_qe/test_qe_knn_token.sh