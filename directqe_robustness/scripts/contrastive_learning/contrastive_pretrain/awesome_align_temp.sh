SRC_ORIGIN_PATH=/data1/yanym/data/robust_word_alignment/qe_data/train.src
TGT_ORIGIN_PATH=/data1/yanym/data/robust_word_alignment/qe_data/train.mt
SRC_BPE_PATH=$SRC_ORIGIN_PATH.BPE
TGT_BPE_PATH=$TGT_ORIGIN_PATH.BPE

MODEL_NAME_OR_PATH=bert-base-multilingual-cased
ALIGN_FILE=/data1/yanym/data/robust_word_alignment/qe_data/train.src-mt.alignments
ALIGN_INPUT_PATH=$ALIGN_FILE.input


# 把src、tgt原始文件用|||合并起来，形成输入对齐工具的文件
python scripts/contrastive_learning/contrastive_pretrain/awesome_align_input.py \
    --src_origin_path $SRC_ORIGIN_PATH \
    --tgt_origin_path $TGT_ORIGIN_PATH \
    --align_input_path $ALIGN_INPUT_PATH
echo "finish construct align input file"

# 词对齐
CUDA_VISIBLE_DEVICES=5 awesome-align \
    --output_file=$ALIGN_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$ALIGN_INPUT_PATH \
    --extraction 'softmax' \
    --batch_size 32
echo "finish alignment"


# bash scripts/contrastive_learning/contrastive_pretrain/awesome_align_temp.sh