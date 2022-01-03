#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_pretrained_256_contrastive_dynamic_from0_111/discriminator.best.ckpt.800" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_finetuned_256_contrastive_dynamic_from0_111_from800" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_contrastive_pretrain/discriminator_finetuned_256_contrastive_dynamic_from0_111_from800" \
    --use_gpu


# bash scripts/contrastive_learning/contrastive_pretrain/finetune_qe.sh 1234