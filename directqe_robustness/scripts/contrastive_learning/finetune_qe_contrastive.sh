#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_contrastive \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "/home/data_ti4_c/yanym/code/directqe_analysis/configs/analysis/contrastive_learning/qe_enzh_wmt20_bpe.yaml" \
    --log_path "/home/yanym/save/directqe_enzh_bpe/discriminator_finetuned_contrastive_256_aug_bi_batch2_lambda20102" \
    --saveto "/home/yanym/save/directqe_enzh_bpe/discriminator_finetuned_contrastive_256_aug_bi_batch2_lambda20102" \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/contrastive_learning/finetune_qe_contrastive.sh 1234