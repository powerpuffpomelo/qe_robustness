#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug_update/discriminator.best.ckpt.87600" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug_2finetune_87600update2normal" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/train_memory/discriminator_finetuned_256_train_memory_data_aug_2finetune_87600update2normal" \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_c_normal.sh 1234