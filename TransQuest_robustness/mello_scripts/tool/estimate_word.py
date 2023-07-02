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
    acc = accuracy_score(goldList, preList)
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
# 注意gold和pred不要写反

# transquest
GOLD_PREFIX=/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_test
PRE_PREFIX=/data1/yanym/save/transquest_wmt20_enzh/test_result
python mello_scripts/estimate_word.py $GOLD_PREFIX/test.mt_tag $PRE_PREFIX/test.mt_tag.pred

# 决策一致性
GOLD_PREFIX=/home/yanym/save/transquest/mt_infl_order/transquest_test_result_normal
PRE_PREFIX=/home/yanym/save/transquest/mt_infl_order/transquest_test_result_3o_sample
python mello_scripts/estimate_word.py $GOLD_PREFIX/tag_src.pred $PRE_PREFIX/tag_src_33.pred

# robust
GOLD_PREFIX=/data1/yanym/data/robust_train_memory
PRE_PREFIX=/data1/yanym/save/transquest_wmt20_enzh/test_result_robust
python mello_scripts/estimate_word.py $GOLD_PREFIX/robust_test.tag_idxonly $PRE_PREFIX/robust_test_train_memory.mt_tag_robust.pred
"""