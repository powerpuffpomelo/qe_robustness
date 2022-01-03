# 构造 对比学习伪数据B-完全正确：基于qe真实数据，使用src + pe；词级全ok，句子级全0
from shutil import copyfile
import random

data_prefix = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train."
save_prefix = "/home/yanym/data/qe_neg_out_of_order/train/train."

src_path = data_prefix + "src"
mt_path = data_prefix + "mt"
pe_path = data_prefix + "pe"
src_save_path = save_prefix + "src"
mt_save_path = save_prefix + "mt"
src_tag_save_path = save_prefix + "src_tag"
mtgap_tag_save_path = save_prefix + "mtgap_tag"
hter_save_path = save_prefix + "hter"

def neg_parallel():
    copyfile(src_path, src_save_path)
    copyfile(mt_path, mt_save_path)

    with open(src_path, 'r', encoding='utf-8') as fsrc, \
        open(mt_path, 'r', encoding='utf-8') as fmt, \
        open(src_tag_save_path, 'w', encoding='utf-8') as fsrc_tag, \
        open(mtgap_tag_save_path, 'w', encoding='utf-8') as fmtgap_tag, \
        open(hter_save_path, 'w', encoding='utf-8') as fhter:

        src_lines = fsrc.readlines()
        mt_lines = fmt.readlines()
        for src_line, mt_line in zip(src_lines, mt_lines):
            src_list = src_line.strip('\n').split()
            mt_list = mt_line.strip('\n').split()
            src_tag_list = ['OK'] * len(src_list)
            mtgap_tag_list = ['OK'] * (len(mt_list) * 2 + 1)
            fsrc_tag.write(' '.join(src_tag_list) + '\n')
            fmtgap_tag.write(' '.join(mtgap_tag_list) + '\n')
            fhter.write('0.0000' + '\n')

# neg_parallel()

# 构造 对比学习伪数据A-完全错误：基于qe真实数据，但打乱src和mt对应关系；词级全bad，句子级全1

seed = 33
random.seed(seed)

def neg_out_of_order():
    copyfile(mt_path, mt_save_path)

    with open(src_path, 'r', encoding='utf-8') as fsrc, \
        open(src_save_path, 'w', encoding='utf-8') as fsrc_save, \
        open(mt_path, 'r', encoding='utf-8') as fmt, \
        open(src_tag_save_path, 'w', encoding='utf-8') as fsrc_tag, \
        open(mtgap_tag_save_path, 'w', encoding='utf-8') as fmtgap_tag, \
        open(hter_save_path, 'w', encoding='utf-8') as fhter:

        src_lines = fsrc.readlines()
        random.shuffle(src_lines)
        for line in src_lines:
            fsrc_save.write(line)
        
        mt_lines = fmt.readlines()
        for src_line, mt_line in zip(src_lines, mt_lines):
            src_list = src_line.strip('\n').split()
            mt_list = mt_line.strip('\n').split()
            src_tag_list = ['BAD'] * len(src_list)
            mtgap_tag_list = ['BAD'] * (len(mt_list) * 2 + 1)
            fsrc_tag.write(' '.join(src_tag_list) + '\n')
            fmtgap_tag.write(' '.join(mtgap_tag_list) + '\n')
            fhter.write('1.0000' + '\n')

neg_out_of_order()

# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/contrastive_learning/data_process.py


"""
BPE_PATH=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/bpe.30000.en
FROM_PATH=/home/yanym/data/qe_neg_out_of_order
TO_PATH=/home/yanym/data/qe_neg_out_of_order
SPLIT=train
subword-nmt apply-bpe -c $BPE_PATH < $FROM_PATH/$SPLIT/$SPLIT.src > $TO_PATH/$SPLIT/$SPLIT.src.BPE


# temp
BPE_PATH=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/bpe.30000.en
FROM_PATH=/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh
TO_PATH=/home/yanym/save
SPLIT=test
subword-nmt apply-bpe -c $BPE_PATH < $FROM_PATH/qe_$SPLIT/$SPLIT.src > $TO_PATH/$SPLIT.src.BPE
"""