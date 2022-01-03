#!/usr/bin/env bash
# finetune时记录loss等信息，以备画图
export CUDA_VISIBLE_DEVICES=5
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_log \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/home/data_ti4_c/yanym/save/enzh/discriminator_pretrained_256_log_v2_complement_loss_record/discriminator.best.final" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --log_path "/home/yanym/save/enzh/discriminator_finetuned_256_log_ckpt_end_end" \
    --saveto "/home/yanym/save/enzh/discriminator_finetuned_256_log_ckpt_end_end" \
    --use_gpu


# bash /home/data_ti4_c/yanym/code/directqe_analysis/scripts/train_directqe_enzh/finetune_qe_log.sh 1234