#!/usr/bin/env bash
# test写入文件，每次都要打开文件太麻烦，这里搞一个直接显示预测结果标签的

export CUDA_VISIBLE_DEVICES=6

# echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_show \
    --pretrain_path "/data1/yanym/data/toy_contrastive/discriminator_normal_small/discriminator.best.ckpt.52" \
    --source_path "/data1/yanym/data/toy_contrastive/temp/generate_dev/mlm_dev.src_temp.BPE" \
    --target_path "/data1/yanym/data/toy_contrastive/temp/generate_dev/mlm_dev.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --batch_size 20 \
    --show_id 0 

# bash scripts/contrastive_learning/contrastive_pretrain/test_qe_show.sh