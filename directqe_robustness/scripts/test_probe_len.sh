#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_probe_len \
    --pretrain_path "./save_qe/discriminator.best.ende.final" \
    --pretrain_path_probe "./save_all/probe_len_pe_hidden/probe_len.best.final" \
    --config_path "./configs/qe_ende_probe.yaml" \
    --log_path "./log_all/log_probe_test_len" \
    --representation "hidden" \
    --len_to_pred "pe" \
    --use_gpu


# sh scripts/test_probe_len.sh > logs/test_probe.log