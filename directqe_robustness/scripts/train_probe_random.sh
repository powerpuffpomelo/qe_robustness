#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_probe \
    --pretrain_path "./save_qe/discriminator.best.ende.final" \
    --config_path "./configs/qe_ende_probe.yaml" \
    --log_path "./log_probe_random" \
    --saveto "./save_probe_random" \
    --representation "random" \
    --use_gpu


# sh scripts/train_probe_random.sh > logs/train_probe_random.log