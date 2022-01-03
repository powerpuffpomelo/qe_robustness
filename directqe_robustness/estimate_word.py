# coding =utf-8
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

def read_File(filePath):
    """Every line corresponds to one hter value"""
    fr = open(filePath, 'r')
    lines = fr.readlines()
    dataList = []
    for line in lines:
        for tag in line.split(' '):
            if 'BAD' in tag:
                dataList.append(0)
            elif 'OK' in tag:
                dataList.append(1)
    return np.array(dataList)

def BPE2word(dataList, text):
    with open(text, 'r', encoding='utf-8') as f:
        text_data = f.read()
    textList = []
    for line in text_data.split('\n'):
        if line == '':
            continue
        line = line.strip()
        for word in line.split(' '):
            textList.append(word)
    preList_new = []

    assert len(textList) == len(dataList)
    i = 0
    while True:
        if i == len(textList):
            break
        if "@@" not in textList[i]:
            preList_new.append(dataList[i])
        else:
            #说明遇到一个subword词
            tag_flag = dataList[i]
            while True:
                if "@@" not in textList[i]:
                    break
                if dataList[i] == 0:
                    tag_flag = 0
                i += 1
            preList_new.append(tag_flag)
        i += 1
    return np.array(preList_new)

if __name__ == '__main__':
    #assert 'home' in sys.argv[1]
    goldList = read_File(sys.argv[1])
    preList = read_File(sys.argv[2])
    #print(preList)
    """
    if 'enzh-word' not in sys.argv[1]:
        preList = BPE2word(preList, sys.argv[3])
    """
    # 测试
    #goldList = [0, 0, 1, 1, 0]
    #preList = [1, 1, 1, 1, 0]

    # 测试结束
    acc = accuracy_score(goldList, preList)
    #print(f1_score(goldList, preList, average=None, pos_label=None))
    #print(f1_score(goldList, preList, average=None, pos_label=None).dtype)
    f1_bad, f1_ok = f1_score(goldList, preList, average=None, pos_label=None)
    precision_bad, precision_ok = precision_score(goldList, preList, average=None)
    recall_bad, recall_ok = recall_score(goldList, preList, average=None)
    print("accuracy = %.4f" % acc)
    print("f1-ok = %.4f" % f1_ok)
    print("f1-bad = %.4f" % f1_bad)
    print("f1-mult = %.4f" % (f1_bad*f1_ok))
    
    print("precision_ok = %.4f" % precision_ok)
    print("precision_bad = %.4f" % precision_bad)
    print("recall_ok = %.4f" % recall_ok)
    print("recall_bad = %.4f" % recall_bad)

"""
# 注意pred和gold不要写反

# 100w enzh
PRE_PREFIX=/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/test_result
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/dev.tag $PRE_PREFIX/dev.tag

# 100w enzh train
PRE_PREFIX=/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_ckpt/test_result_ckpt
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/train.tag $PRE_PREFIX/train.tag.34800

# enzh
PRE_PREFIX=/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/test_result
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/train.tag $PRE_PREFIX/train.tag 
python estimate_word.py $GOLD_PREFIX/dev.tag $PRE_PREFIX/dev.tag 
python estimate_word.py $GOLD_PREFIX/test.tag $PRE_PREFIX/test.tag

# with_parallel_cl ende
PRE_PREFIX=/home/data_ti4_c/yanym/save/ende/discriminator_finetuned_256_with_parallel_cl_aug50/test_result
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version
python estimate_word.py $GOLD_PREFIX/test.tag.BPE $PRE_PREFIX/test.tag

# src 3o
PRE_PREFIX=/home/yanym/save/enzh/src_infl/order/test_result_3o_sample
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version
python estimate_word.py $GOLD_PREFIX/test.tag $PRE_PREFIX/test.tag_33

# 决策一致性
PRE_PREFIX=/home/yanym/save/enzh/src_infl/order/test_result_3o_sample
GOLD_PREFIX=/home/yanym/save/enzh/discriminator_finetuned_aug_shuffle/test_result
python estimate_word.py $GOLD_PREFIX/test.tag $PRE_PREFIX/test.tag_33

# transquest
PRE_PREFIX=/home/yanym/save/enzh/src_infl/order/transquest_train_result_3o_sample
GOLD_PREFIX=/home/yanym/save/enzh/src_infl/order/transquest_train_result_3o_sample
python estimate_word.py $GOLD_PREFIX/predictions_mt_normal.txt $PRE_PREFIX/predictions_mt_3o_sample_33.txt

# wmt21 enzh
PRE_PREFIX=/home/yanym/save/enzh/discriminator_finetuned_256_v4_wmt21/test_result_ckpt12000
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test
python estimate_word.py $GOLD_PREFIX/test.mt_tag $PRE_PREFIX/test.tag
python estimate_sentence.py $GOLD_PREFIX/test.hter $PRE_PREFIX/test.hter

"""