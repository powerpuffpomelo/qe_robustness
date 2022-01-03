#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/ende/discriminator_finetuned_256/discriminator.best.final" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/analysis/gradient_analysis/qe_ende.yaml" \
    --plot_path_prefix "/data1/yanym/save/ende/gradient/" \
    --align_path_prefix "/data1/yanym/save/ende/gradient/" \
    --align_bpe_path_src2mt "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version_transquest/qe_test/test.src-mt.alignments.BPE" \
    --align_bpe_path_mt2src "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version_transquest/qe_test/test.mt-src.alignments.BPE" \
    --gradients_type "g" \
    --test_id 0 \
    --ig_steps 100 \
    --batch_size 20 \
    --use_gpu 

# bash scripts/analysis/gradient_analysis/attribution_qe_ende_wmt20.sh