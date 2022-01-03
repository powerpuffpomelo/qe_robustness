SRC_ORIGIN_PATH=/data1/yanym/data/robust_qe_test/generate_dev/robust_test.mt
TGT_ORIGIN_PATH=/data1/yanym/data/robust_qe_test/generate_dev/robust_test.src
SRC_BPE_PATH=$SRC_ORIGIN_PATH.BPE
TGT_BPE_PATH=$TGT_ORIGIN_PATH.BPE

MODEL_NAME_OR_PATH=bert-base-multilingual-cased
ALIGN_FILE=/data1/yanym/data/robust_qe_test/generate_dev/robust_test.mt-src.alignments
ALIGN_INPUT_PATH=$ALIGN_FILE.input
ALIGN_FILE_BPE=$ALIGN_FILE.BPE


# 把src、tgt原始文件用|||合并起来，形成输入对齐工具的文件
python scripts/contrastive_learning/contrastive_pretrain/awesome_align_input.py \
    --src_origin_path $SRC_ORIGIN_PATH \
    --tgt_origin_path $TGT_ORIGIN_PATH \
    --align_input_path $ALIGN_INPUT_PATH
echo "finish construct align input file"

# 词对齐
CUDA_VISIBLE_DEVICES=3 awesome-align \
    --output_file=$ALIGN_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$ALIGN_INPUT_PATH \
    --extraction 'softmax' \
    --batch_size 32
echo "finish alignment"

# 词对齐bpe（单向，维护一个src token对齐的全部tgt token
python scripts/analysis/gradient_analysis/align_file2bpe.py \
    --align_gold_file $ALIGN_FILE \
    --src_origin_file $SRC_ORIGIN_PATH \
    --src_bpe_file $SRC_BPE_PATH \
    --tgt_origin_file $TGT_ORIGIN_PATH \
    --tgt_bpe_file $TGT_BPE_PATH \
    --align_gold_bpe_file $ALIGN_FILE_BPE
echo "finish alignment bpe"



# bash scripts/contrastive_learning/contrastive_pretrain/awesome_align_temp.sh