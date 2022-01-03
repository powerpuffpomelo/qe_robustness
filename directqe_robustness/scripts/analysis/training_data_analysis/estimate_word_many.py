# coding =utf-8
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

train_num_threshold = 5     # 注意这里

vocab = []
vocab_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/vocab.zh"
with open(vocab_path, 'r', encoding='utf-8') as fv:
    for line in fv.readlines():
        line = line.strip('\n').split()
        vocab.append(line[0])

train_mt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt"
dev_mt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/dev.mt"
test_mt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt"
#pred_train_prefix = "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many_" + str(train_num_threshold) + "/" 
pred_train_prefix = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/"

cnt_train = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/stat_info/qe_train_token_cnt"
gold_train = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.tag"
pred_train = pred_train_prefix + "test_result/train.tag"

cnt_dev = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/stat_info/qe_dev_token_cnt"
gold_dev = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/dev.tag"
pred_dev = pred_train_prefix + "test_result/dev.tag"

cnt_test = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/stat_info/qe_test_token_cnt"
gold_test = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag"
pred_test = pred_train_prefix + "test_result/test.tag"

cnt_test_ckpt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/stat_info/qe_test_token_cnt"
gold_test_ckpt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag"
pred_test_ckpt = pred_train_prefix + "test_result_ckpt/test.tag.12000"

cnt_test_begin = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/stat_info/qe_test_token_cnt"
gold_test_begin = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag"
pred_test_begin = pred_train_prefix + "test_result_begin/test.tag"

def estimate(train_num_threshold, test_mt, cnt_test, gold_test, pred_test):
    goldList = []
    preList = []

    with open(gold_test, 'r', encoding='utf-8') as fg, open(pred_test, 'r', encoding='utf-8') as fp, \
        open(cnt_test, 'r', encoding='utf-8') as fc, open(test_mt, 'r', encoding='utf-8') as ft:
        gold_lines = fg.readlines()
        pred_lines = fp.readlines()
        cnt_lines = fc.readlines()
        test_lines = ft.readlines()
        for gl, pl, cl, tl in zip(gold_lines, pred_lines, cnt_lines, test_lines):
            gl = gl.strip('\n').split()
            pl = pl.strip('\n').split()
            cl = cl.strip('\n').split()
            tl = tl.strip('\n').split()
            for g, p, c, t in zip(gl, pl, cl, tl):
                if int(c) < train_num_threshold or t not in vocab: continue
                if g == "BAD": goldList.append(0)
                elif g == "OK": goldList.append(1)
                if p == "BAD": preList.append(0)
                elif p == "OK": preList.append(1)

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

#"""
print("================ train")
estimate(train_num_threshold, train_mt, cnt_train, gold_train, pred_train)
print("================ dev")
estimate(train_num_threshold, dev_mt, cnt_dev, gold_dev, pred_dev)
print("================ test")
estimate(train_num_threshold, test_mt, cnt_test, gold_test, pred_test)
#"""

#estimate(train_num_threshold, test_mt, cnt_test_ckpt, gold_test_ckpt, pred_test_ckpt)
#estimate(train_num_threshold, test_mt, cnt_test_begin, gold_test_begin, pred_test_begin)

# python scripts/analysis/training_data_analysis/estimate_word_many.py 
