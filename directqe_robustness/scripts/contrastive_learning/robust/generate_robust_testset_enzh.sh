#!/usr/bin/env bash
# 每条数据延展出修改一点点的好几条伪数据

export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python3 -m src.bin.generate_dev_src_robust \
    --model_name "generator_mlm" \
    --lm MLM \
    --config_path "./configs/contrastive/generate_dev_enzh_robust.yaml" \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe_contrastive_pretrain/generator_wwm/generator_mlm.best.final" \
    --log_path "/data1/yanym/data/robust_pretrain_v1/generate_dev" \
    --saveto "/data1/yanym/data/robust_pretrain_v1/generate_dev/" \
    --align_bpe_path_mt2src "/data1/yanym/data/robust_pretrain_v1/pseudo_train_v2.mt-src.alignments.BPE" \
    --split "train" \
    --use_gpu

# bash scripts/contrastive_learning/robust/generate_robust_testset_enzh.sh