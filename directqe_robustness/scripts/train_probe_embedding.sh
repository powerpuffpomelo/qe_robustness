#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_probe \
    --pretrain_path "./save_qe/discriminator.best.ende.final" \
    --config_path "./configs/qe_ende_probe.yaml" \
    --log_path "./log_probe_embedding" \
    --saveto "./save_probe_embedding" \
    --representation "embedding" \
    --use_gpu


# sh scripts/train_probe_embedding.sh > logs/train_probe_embedding.log