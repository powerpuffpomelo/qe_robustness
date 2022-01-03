#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_knn_hter \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256/discriminator.best.final" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/analysis/knn_analysis/qe_enzh_knn.yaml" \
    --batch_size 20 \
    --k 20 \
    --temperature 0.1 \
    --lambda_knn 1 \
    --faiss_index_prefix "/home/data_ti4_c/yanym/save/enzh/knn/faiss_index/" \
    --saveto_hter "/home/yanym/save/enzh/knn_predict/test_k20_t0.1_lambda1.hter" \
    --saveto_tags "/home/yanym/save/enzh/knn_predict/test.tag" \
    --use_gpu

# bash scripts/knn_qe/test_qe_knn_hter.sh