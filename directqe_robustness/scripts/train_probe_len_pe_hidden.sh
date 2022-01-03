#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_probe_len \
    --pretrain_path "./save_qe/discriminator.best.ende.final" \
    --config_path "./configs/qe_ende_probe.yaml" \
    --log_path "./log_all/probe_len_pe_hidden" \
    --saveto "./save_all/probe_len_pe_hidden" \
    --representation "hidden" \
    --len_to_pred "pe" \
    --use_gpu


# sh scripts/train_probe_len_pe_hidden.sh