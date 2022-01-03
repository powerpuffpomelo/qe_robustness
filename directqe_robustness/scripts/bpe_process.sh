NUM=30000
BPE_NAME=bpe.$NUM.zh
VOCABDIR=/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version
DIR=/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_dev
TO=/data1/yanym/data

#echo "Learn BPE."
#subword-nmt learn-bpe -s $NUM -o $DIR/$BPE_NAME < $DIR/train.enzh.zh.plus_qe_mt

#echo "Get vocabs."
#subword-nmt apply-bpe -c $DIR/$BPE_NAME < $DIR/train.enzh.zh.plus_qe_mt | subword-nmt get-vocab > $DIR/vocab.zh


for SPLIT in "dev.mt" ; do
  subword-nmt apply-bpe -c $VOCABDIR/$BPE_NAME --vocabulary $VOCABDIR/vocab.zh --vocabulary-threshold 50 < $DIR/$SPLIT > $TO/$SPLIT.BPE
done

# bash scripts/bpe_process.sh