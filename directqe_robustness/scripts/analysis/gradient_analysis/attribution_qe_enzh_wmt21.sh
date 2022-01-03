#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.attribution_qe \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200_wmt21/discriminator.best.ckpt.18000" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/analysis/gradient_analysis/qe_enzh_wmt21.yaml" \
    --plot_path_prefix "/data1/yanym/save/directqe_enzh_bpe/gradient_wmt21/" \
    --align_bpe_path_mt2src "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.mt-src.alignments.BPE" \
    --acc_sign_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200_wmt21/test_result_ckpt18000/test.tag_acc.sign" \
    --gradients_type "g" \
    --test_id 0 \
    --ig_steps 100 \
    --batch_size 20 \
    --use_gpu 

# bash scripts/analysis/gradient_analysis/attribution_qe_enzh_wmt21.sh