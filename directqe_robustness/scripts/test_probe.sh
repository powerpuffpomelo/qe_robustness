#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_probe \
    --pretrain_path "./save_qe/discriminator.best.ende.final" \
    --pretrain_path_probe "./save_probe/probe.best.final" \
    --config_path "./configs/qe_ende_probe.yaml" \
    --log_path "./log_probe_test" \
    --representation "hidden" \
    --use_gpu


# sh scripts/test_probe.sh > logs/test_probe.log