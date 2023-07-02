#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=6

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_ensemble \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE" \
    --config_path "./configs/robust/qe_enzh_ensemble.yaml" \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/save/directqe_enzh_bpe_robust/boosting/discriminator_finetuned_256_boosting_from_pretrained/test_result_18000_boosting/test.hter" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/boosting/discriminator_finetuned_256_boosting_from_pretrained/test_result_18000_boosting/test.tag.BPE" \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/boosting/discriminator_finetuned_256_boosting_from_pretrained/test_result_18000_boosting
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/test.tag.BPE $PRE_PREFIX/test.tag.BPE
python estimate_sentence.py $PRE_PREFIX/test.hter $GOLD_PREFIX/test.hter

# bash scripts/contrastive_learning/robust/test_qe_ensemble.sh