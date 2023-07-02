NUM=30000
LANG=en
BPE_NAME=bpe.$NUM.$LANG
VOCABDIR=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/vocab
DIR=/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp
TO=/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp

#echo "Learn BPE."
#subword-nmt learn-bpe -s $NUM -o $DIR/$BPE_NAME < $DIR/test.enzh.zh.plus_qe_mt

#echo "Get vocabs."
#subword-nmt apply-bpe -c $DIR/$BPE_NAME < $DIR/test.enzh.zh.plus_qe_mt | subword-nmt get-vocab > $DIR/vocab.zh


for SPLIT in "robust_train.src"; do
  subword-nmt apply-bpe -c $VOCABDIR/$BPE_NAME --vocabulary $VOCABDIR/vocab.$LANG --vocabulary-threshold 50 < $DIR/$SPLIT > $TO/$SPLIT.BPE
done

# bash scripts/bpe_process.sh