#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_token_mask_every_sample \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/directqe_enzh_bpe/discriminator_pretrained_256_v4_worse_pseudo_mt_temp/discriminator.best.ckpt.11200" \
    --config_path "./configs/robust/qe_enzh_aug_wmt20_train_memory_every_sample.yaml" \
    --sample_seg_path "/data1/yanym/data/robust_train_memory/data_augmentation_train_update/robust_train.sample_seg.json" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug_every_sample" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug_every_sample" \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_train_memory_data_aug_every_sample.sh 1234