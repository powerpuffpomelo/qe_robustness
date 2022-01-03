# coding =utf-8
# qe质量与词频的关联

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import string
import json
import re
from scipy.stats import pearsonr
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# enzh
data_prefix = "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/"
train_mt_file = "train.mt.BPE"
test_mt_file = "test.mt.BPE"
test_tag_pred_file = "result/test.tag_pre"
test_tag_gold_file = "test.tags.BPE"

# ende
"""
data_prefix = "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/"
train_mt_file = "train.mt.BPE"
test_mt_file = "test.mt.BPE"
test_tag_pred_file = "result/test.tag_pre"
test_tag_gold_file = "test.tags.BPE"
"""

# 过一遍train mt，得到train词表集合
train_word_freq = dict()
with open(data_prefix+train_mt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.lower()
        tokens = re.split(r"[ ]+", line.strip('\n'))
        if tokens == []: continue
        for w in tokens:
            #w = w.strip(string.punctuation)
            if w == '': continue
            if w not in train_word_freq:
                train_word_freq[w] = 0
            train_word_freq[w] += 1

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
                test_token_stat[token] = {"seen":False, "freq":0, "pre_list":[], "gold_list":[]}
                if token in train_word_freq:
                    test_token_stat[token]["seen"] = True
            if line_pred[index] == 'OK':
                test_token_stat[token]["pre_list"].append(1)  # ok1 bad0
            else:
                test_token_stat[token]["pre_list"].append(0)
            if line_gold[index] == 'OK':
                test_token_stat[token]["gold_list"].append(1)
            else:
                test_token_stat[token]["gold_list"].append(0)

freq_acc_stat = dict()  # 键是训练集freq，值是dict，包括 1test_num 2acc加和 3acc_avg

for token, stat in test_token_stat.items():
    if token in train_word_freq:
        stat["freq"] = train_word_freq[token]
    else:
        stat["freq"] = 0
    acc = accuracy_score(stat["gold_list"], stat["pre_list"])  # 一个词的acc
    stat["acc"] = acc
    if stat["freq"] not in freq_acc_stat:
        freq_acc_stat[stat["freq"]] = {"gold_list":[], "pre_list":[]}
    freq_acc_stat[stat["freq"]]["gold_list"].extend(stat["gold_list"])
    freq_acc_stat[stat["freq"]]["pre_list"].extend(stat["pre_list"])


# ===================================================================================== # 
# 词频和acc相关性

def cal_info(x,y):
    pearson_corrcoef = pearsonr(x, y)[0]
    mi = metrics.mutual_info_score(x,y)
    nmi = metrics.normalized_mutual_info_score(x,y)
    ami = metrics.adjusted_mutual_info_score(x,y)
    print("%.4f" % pearson_corrcoef)
    print("%.4f" % mi)
    print("%.4f" % nmi)
    print("%.4f" % ami)
    return pearson_corrcoef, mi, nmi, ami


freq_list = [test_token_stat[token]["freq"] for token in test_token_stat]  # 长度就是所有token
acc_list = [test_token_stat[token]["acc"] for token in test_token_stat]
print("cal info freq acc")
print(len(freq_list))
#print(freq_list)
#print(acc_list)
cal_info(freq_list, acc_list)


# ===================================================================================== # 
# 词频和acc分段讨论,freq合并画图，优化上边的图

freq_list = []
acc_list = []
for freq,stat in freq_acc_stat.items():
    freq_list.append(freq)
    acc_list.append(accuracy_score(stat["gold_list"], stat["pre_list"]))

print("==============================")
print(len(freq_list))
print(freq_list[:10])

merge_list = list(zip(freq_list, acc_list))
merge_list_sorted = sorted(merge_list, key=lambda x:x[0])
new_freq, new_acc = zip(*merge_list_sorted)
#print(new_freq)
#print(new_acc)




# 合并相同freq后token画图
f, ax = plt.subplots(figsize = (5,5))
ax.plot(new_freq, new_acc)

ax.set_xlabel('freq')
#ax.set_xticklabels(new_freq, rotation=90)
ax.set_ylabel('acc')

plot_path_prefix = "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/result/"
plt.savefig(plot_path_prefix + "freq_acc_merge.jpg")

"""
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(x, x, label='linear')  # Plot some data on the axes.
ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
ax.plot(x, x**3, label='cubic')  # ... and some more.
ax.set_xlabel('x label')  # Add an x-label to the axes.
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.
"""

# python scripts/attribution_enzh/freq_plot.py