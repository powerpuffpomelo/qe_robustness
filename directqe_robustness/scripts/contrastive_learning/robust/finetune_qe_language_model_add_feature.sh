#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.finetune_qe_token_mask \
    --model_name "discriminator" \
    --qe_model_name "qe" \
    --reload \
    --seed $1 \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/add_feature/discriminator_finetuned_256_add_feature/discriminator.best.ckpt.18000" \
    --config_path "./configs/robust/qe_enzh_aug_wmt20_language_model.yaml" \
    --add_feature \
    --align_bpe_path_mt2src_train "/data1/yanym/data/robust_language_model/robust_qe_train_word2vec_double/align/robust_train.mt-src.alignments.BPE.json" \
    --align_bpe_path_mt2src_dev "/data1/yanym/data/test_align_enzh/dev.mt-src.alignments.BPE" \
    --log_path "/data1/yanym/save/directqe_enzh_bpe_robust/language_model/discriminator_finetuned_256_language_model_add_feature_n2r" \
    --saveto "/data1/yanym/save/directqe_enzh_bpe_robust/language_model/discriminator_finetuned_256_language_model_add_feature_n2r" \
    --use_gpu


# bash scripts/contrastive_learning/robust/finetune_qe_language_model_add_feature.sh 1234