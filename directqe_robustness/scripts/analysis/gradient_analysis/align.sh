SRC_ORIGIN_PATH=/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.mt
TGT_ORIGIN_PATH=/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.src
ALIGN_FILE=/data1/yanym/data/test.mt-src.alignments

SRC_BPE_PATH=$SRC_ORIGIN_PATH.BPE
TGT_BPE_PATH=$TGT_ORIGIN_PATH.BPE
ALIGN_FILE_BPE=$ALIGN_FILE.BPE


# 词对齐bpe（单向，维护一个src token对齐的全部tgt token
python scripts/analysis/gradient_analysis/align_file2bpe.py \
    --align_gold_file $ALIGN_FILE \
    --src_origin_file $SRC_ORIGIN_PATH \
    --src_bpe_file $SRC_BPE_PATH \
    --tgt_origin_file $TGT_ORIGIN_PATH \
    --tgt_bpe_file $TGT_BPE_PATH \
    --align_gold_bpe_file $ALIGN_FILE_BPE
echo "finish alignment bpe"



# bash scripts/analysis/gradient_analysis/align.sh