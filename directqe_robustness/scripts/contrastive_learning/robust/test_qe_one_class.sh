#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_qe_one_class \
    --pretrain_path "/data1/yanym/save/directqe_enzh_bpe_robust/one_class/discriminator_finetuned_256_one_class_from13120/discriminator.best.ckpt.8400" \
    --source_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE" \
    --target_path "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE" \
    --config_path "./configs/train_directqe_enzh/qe_enzh_one_class.yaml" \
    --batch_size 20 \
    --saveto_hter "/data1/yanym/save/directqe_enzh_bpe_robust/one_class/discriminator_finetuned_256_one_class_from13120/test_result_8400/test.hter" \
    --saveto_tags "/data1/yanym/save/directqe_enzh_bpe_robust/one_class/discriminator_finetuned_256_one_class_from13120/test_result_8400/test.tag.BPE" \
    --use_gpu


# test result
PRE_PREFIX=/data1/yanym/save/directqe_enzh_bpe_robust/one_class/discriminator_finetuned_256_one_class_from13120/test_result_8400
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/test.tag.BPE $PRE_PREFIX/test.tag.BPE
python estimate_sentence.py $PRE_PREFIX/test.hter $GOLD_PREFIX/test.hter

# bash scripts/contrastive_learning/robust/test_qe_one_class.sh