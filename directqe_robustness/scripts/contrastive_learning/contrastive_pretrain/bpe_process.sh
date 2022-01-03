NUM=30000
BPE_NAME=bpe.$NUM.en
VOCABDIR=/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version
DIR=/data1/yanym/data/toy_contrastive/data_toy2
TO=/data1/yanym/data/toy_contrastive/data_toy2

#echo "Learn BPE."
#subword-nmt learn-bpe -s $NUM -o $DIR/$BPE_NAME < $DIR/train.enzh.zh.plus_qe_mt

#echo "Get vocabs."
#subword-nmt apply-bpe -c $DIR/$BPE_NAME < $DIR/train.enzh.zh.plus_qe_mt | subword-nmt get-vocab > $DIR/vocab.zh


for SPLIT in "toy_train.src" ; do
  subword-nmt apply-bpe -c $VOCABDIR/$BPE_NAME --vocabulary $VOCABDIR/vocab.en --vocabulary-threshold 50 < $DIR/$SPLIT > $TO/$SPLIT.BPE
done

# bash scripts/contrastive_learning/contrastive_pretrain/bpe_process.sh