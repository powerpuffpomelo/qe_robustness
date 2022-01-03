# coding =utf-8
# 比较测试集中【在训练集见过的】【没见过的】token的标注情况
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import string
import json
import re
"""
print("======================================")
gold_list = np.array([1]*10 + [0]*10)
pre_list = np.array([1]*10 + [0]*10)   # 跟这个类型没关系，这个是int，不影响f1是float
print(pre_list)
ff = f1_score(gold_list, pre_list, average=None, pos_label=None)
print(ff)
print(ff.dtype)

f1_bad, f1_ok = f1_score(gold_list, pre_list, average=None, pos_label=None)
f1_bad, f1_ok = f1_score(gold_list, pre_list)
assert 1==2
"""

data_prefix = "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/"
train_mt_file = "train.mt"
test_mt_file = "test.mt"
test_tag_pred_file = "result/test.tag_pre"
test_tag_gold_file = "test.tag"

# 过一遍train mt，得到train词表集合
train_wordset = set()
with open(data_prefix+train_mt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.lower()
        tokens = re.split(r"[ ]+", line.strip('\n'))
        if tokens == []: continue
        for w in tokens:
            #w = w.strip(string.punctuation)
            if w == '': continue
            train_wordset.add(w)

# 过一遍test mt，直接得到test token字典：不在键中就把词加入；并判断是否在训练词表中，得到seen布尔值 goldlist prelist；
test_token_stat = dict()

with open(data_prefix+test_mt_file, 'r', encoding='utf-8') as f_test_mt, \
    open(data_prefix+test_tag_pred_file, 'r', encoding='utf-8') as f_test_tag_pred, \
    open(data_prefix+test_tag_gold_file, 'r', encoding='utf-8') as f_test_tag_gold:
    for line_mt, line_pred, line_gold in zip(f_test_mt, f_test_tag_pred, f_test_tag_gold):
        line_mt = line_mt.lower()
        tokens = re.split(r"[ ]+", line_mt.strip('\n'))
        line_pred = line_pred.strip('\n').split()
        line_gold = line_gold.strip('\n').split()
        assert len(tokens) == len(line_pred) == len(line_gold)
        #print(tokens)
        for index,token in enumerate(tokens):
            if token not in test_token_stat:
                test_token_stat[token] = {"seen":False, "pre_list":[], "gold_list":[]}
                if token in train_wordset:
                    test_token_stat[token]["seen"] = True
            if line_pred[index] == 'OK':
                test_token_stat[token]["pre_list"].append(1)  # ok1 bad0
            else:
                test_token_stat[token]["pre_list"].append(0)
            if line_gold[index] == 'OK':
                test_token_stat[token]["gold_list"].append(1)
            else:
                test_token_stat[token]["gold_list"].append(0)

seen_pred = []
seen_gold = []
unseen_pred = []
unseen_gold = []
seen_num = 0
unseen_num = 0
freq_acc_stat = dict()

for token, stat in test_token_stat.items():
    pre_list = test_token_stat[token]["pre_list"]
    gold_list = test_token_stat[token]["gold_list"]
    acc = accuracy_score(gold_list, pre_list)
    
    #print(f1_score([0,1], [0,0], average=None, pos_label=None))
    #TODO f1_bad, f1_ok = f1_score(gold_list, pre_list, average=None, pos_label=None)
    f1_bad, f1_ok = 0, 0
    test_token_stat[token]["eval"] = {"acc":acc, "f1_bad":f1_bad, "f1_ok":f1_ok, "f1_multi":f1_bad*f1_ok}

    if test_token_stat[token]["seen"] == True:
        seen_num += 1
        seen_eval["acc"] += test_token_stat[token]["eval"]["acc"]
        seen_eval["f1_bad"] += test_token_stat[token]["eval"]["f1_bad"]
        seen_eval["f1_ok"] += test_token_stat[token]["eval"]["f1_ok"]
        seen_eval["f1_multi"] += test_token_stat[token]["eval"]["f1_multi"]
    else:
        unseen_num += 1
        unseen_eval["acc"] += test_token_stat[token]["eval"]["acc"]
        unseen_eval["f1_bad"] += test_token_stat[token]["eval"]["f1_bad"]
        unseen_eval["f1_ok"] += test_token_stat[token]["eval"]["f1_ok"]
        unseen_eval["f1_multi"] += test_token_stat[token]["eval"]["f1_multi"]

seen_eval["acc"] = seen_eval["acc"] / seen_num
unseen_eval["acc"] = unseen_eval["acc"] / unseen_num

print("aaaaaaaaaaaaaaaaaa")
print(seen_num)
print(unseen_num)
print(seen_eval)
print(unseen_eval)

# python scripts/attribution_enzh/seen_vs_unseen_token.py