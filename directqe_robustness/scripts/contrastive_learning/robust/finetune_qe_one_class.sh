#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=6
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_one_class \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/one_class/discriminator_pretrained_256_one_class/discriminator.best.ckpt.13120" \
    --config_path "./configs/train_directqe_enzh/qe_enzh_one_class.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/one_class/discriminator_finetuned_256_one_class_from13120" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/one_class/discriminator_finetuned_256_one_class_from13120" \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_one_class.sh 1234