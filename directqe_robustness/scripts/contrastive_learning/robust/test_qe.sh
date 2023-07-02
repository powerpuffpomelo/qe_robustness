#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/language_model/discriminator_finetuned_256_language_model_add_feature_n2r/discriminator.best.ckpt.58800" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh.yaml" \
    --add_feature \
    --align_bpe_path_mt2src_test "/data1/yanym/data/test_align_enzh/test.mt-src.alignments.BPE.json" \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/save/directqe_enzh_bpe_robust/language_model/discriminator_finetuned_256_language_model_add_feature_n2r/test_result_58800/test.hter" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/language_model/discriminator_finetuned_256_language_model_add_feature_n2r/test_result_58800/test.tag.BPE" \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/language_model/discriminator_finetuned_256_language_model_add_feature_n2r/test_result_58800
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/test.tag.BPE $PRE_PREFIX/test.tag.BPE
python estimate_sentence.py $PRE_PREFIX/test.hter $GOLD_PREFIX/test.hter

# bash scripts/contrastive_learning/robust/test_qe.sh