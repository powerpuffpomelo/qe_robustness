"""
完全极端统计
"""

import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math

vocab_zh_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/vocab.zh"
vocab_zh_list = []
with open(vocab_zh_path, 'r', encoding='utf-8') as f_v:
    lines = f_v.readlines()
    for line in lines:
        token = line.strip('\n').split()[0]
        vocab_zh_list.append(token)

qe_test_mt_path = "/home/yanym/save/enzh/vocab/test.mt"
#qe_test_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_ckpt/test_result_ckpt/test.tag.22800"
qe_test_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/test_result/test.tag"
#qe_test_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_ckpt_end_end/test_result/test.tag"
qe_test_gold_tag_path = "/home/yanym/save/enzh/vocab/test.tag_gold"

qe_train_mt_path = "/home/yanym/save/enzh/vocab/train.mt"
#qe_train_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_ckpt/test_result_ckpt/train.tag.22800"
qe_train_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/test_result/train.tag"
#qe_train_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_ckpt_end_end/test_result/train.tag"
qe_train_gold_tag_path = "/home/yanym/save/enzh/vocab/train.tag_gold"

qe_train_vocab_path = "/home/yanym/save/enzh/vocab/qe_train.mt.json"
save_path = "/home/yanym/save/enzh/stat/qe_test_token_stat_train.json"

# 构造token大字典，仅限于test和qe_train交集的token，把这些token的测试和训练时的gold/pred tag都存下来
def make_stat_dict(test_mt, test_pred, test_gold, train_mt, train_pred, train_gold, qe_train_vocab_path, save_path):
    ret_dict = dict()
    with open(qe_train_vocab_path, 'r', encoding='utf-8') as f_q:
        qe_train_vocab = json.load(f_q)
    with open(test_mt, 'r', encoding='utf-8') as f_mt, open(test_pred, 'r', encoding='utf-8') as f_pred, \
        open(test_gold, 'r', encoding='utf-8') as f_gold:
        for line_mt, line_pred, line_gold in zip(f_mt.readlines(), f_pred.readlines(), f_gold.readlines()):
            line_mt = line_mt.strip('\n').split()
            line_pred = line_pred.strip('\n').split()
            line_gold = line_gold.strip('\n').split()
            for i, token in enumerate(line_mt):
                if token in qe_train_vocab:
                    if token not in ret_dict:
                        ret_dict[token] = {'test_freq':0, 'test_tag_pred':[], 'test_tag_gold':[], 'test_acc':0, \
                                            'train_freq':0, 'train_tag_pred':[], 'train_tag_gold':[], 'train_acc':0}
                    ret_dict[token]['test_freq'] += 1
                    ret_dict[token]['test_tag_pred'].append(line_pred[i])
                    ret_dict[token]['test_tag_gold'].append(line_gold[i])
    with open(train_mt, 'r', encoding='utf-8') as f_mt, open(train_pred, 'r', encoding='utf-8') as f_pred, \
        open(train_gold, 'r', encoding='utf-8') as f_gold:
        for line_mt, line_pred, line_gold in zip(f_mt.readlines(), f_pred.readlines(), f_gold.readlines()):
            line_mt = line_mt.strip('\n').split()
            line_pred = line_pred.strip('\n').split()
            line_gold = line_gold.strip('\n').split()
            for i, token in enumerate(line_mt):
                if token not in ret_dict: continue
                ret_dict[token]['train_freq'] += 1
                #print(line_pred)
                #print(line_gold)
                ret_dict[token]['train_tag_pred'].append(line_pred[i])
                ret_dict[token]['train_tag_gold'].append(line_gold[i])
    for token, stat in ret_dict.items():
        stat['test_acc'] = accuracy_score(stat['test_tag_pred'], stat['test_tag_gold'])
        stat['train_acc'] = accuracy_score(stat['train_tag_pred'], stat['train_tag_gold'])
    with open(save_path, 'w', encoding='utf-8') as f_s:
        json.dump(ret_dict, f_s, indent=1)

#"""
make_stat_dict(test_mt=qe_test_mt_path,
                test_pred=qe_test_pred_tag_path,
                test_gold=qe_test_gold_tag_path,
                train_mt=qe_train_mt_path,
                train_pred=qe_train_pred_tag_path,
                train_gold=qe_train_gold_tag_path,
                qe_train_vocab_path=qe_train_vocab_path,
                save_path=save_path)
#"""

# 比例
train_gold_ok_ratio_list = []
train_pred_ok_ratio_list = []
test_gold_ok_ratio_list = []
test_pred_ok_ratio_list = []

# 倾向
test_pred_ok_ratio_list_tendency = []
train_pred_ok_ratio_list_tendency = []
train_gold_ok_ratio_list_tendency = []
# 极端
extreme_cnt = 0
same_cnt = 0
anti_extreme_cnt = 0
all_cnt = 0
# 画图
plt_test_pred = []
plt_train_gold = []
plt_test_pred_dict = dict()
plt_train_gold_dict = dict()

# 不极端个案
train_gold_ok_0_test_pred_ok_1 = []
train_gold_ok_1_test_pred_ok_0 = []

num_threshold = 100

token_cnt_wo_unk = 0   # 字典里不是unk的有多少个token

freq_threshold = 10
freq_cnt = 0
extreme_cnt_1 = 0
extreme_cnt_2 = 0
extreme_cnt_3 = 0
extreme_cnt_23 = 0
anti_extreme = []
extreme_stat_dict = dict()
extreme_stat_path = "/home/yanym/save/enzh/stat/extreme_stat_5.json"
train_pred_extreme_cnt = 0
train_freq_cnt = 0
train_pred_extreme_poor_cnt = 0
train_pred_extreme_poor_dict = dict()
with open(save_path, 'r', encoding='utf-8') as f_stat:
    token_stat = json.load(f_stat)
    #print(len(token_stat))
    #assert 1==2
    for token, stat in token_stat.items():
        num_test_tag_pred = [1 if tag == 'OK' else 0 for tag in stat['test_tag_pred']]
        num_test_tag_gold = [1 if tag == 'OK' else 0 for tag in stat['test_tag_gold']]
        num_train_tag_pred = [1 if tag == 'OK' else 0 for tag in stat['train_tag_pred']]
        num_train_tag_gold = [1 if tag == 'OK' else 0 for tag in stat['train_tag_gold']]
        length_test = len(num_test_tag_pred)
        length_train = len(num_train_tag_pred)
        test_pred_ok_ratio = sum(num_test_tag_pred) / length_test
        test_gold_ok_ratio = sum(num_test_tag_gold) / length_test
        train_pred_ok_ratio = sum(num_train_tag_pred) / length_train
        train_gold_ok_ratio = sum(num_train_tag_gold) / length_train

        if token in vocab_zh_list:
            token_cnt_wo_unk += 1
            if stat['test_freq'] >= freq_threshold:
                freq_cnt += 1
                if test_pred_ok_ratio == 0 or test_pred_ok_ratio == 1:        # test_pred完全极端化
                    extreme_cnt_1 += 1
                    if test_gold_ok_ratio != test_pred_ok_ratio:              # test_gold不应极端化
                        extreme_cnt_2 += 1
                    if abs(train_gold_ok_ratio - test_pred_ok_ratio) < 0.5:   # test_pred和train_gold方向一致
                        extreme_cnt_3 += 1
                        if test_gold_ok_ratio != test_pred_ok_ratio:
                            extreme_cnt_23 += 1
                            extreme_stat_dict[token] = stat
                    else:
                        anti_extreme.append(token)
            if train_pred_ok_ratio == 0 or train_pred_ok_ratio == 1:        # train_pred完全极端化
                train_pred_extreme_cnt += 1
                if stat['train_freq'] >= freq_threshold:              # train_pred 超过阈值的个数
                    train_freq_cnt += 1
                    if train_gold_ok_ratio != train_pred_ok_ratio:    # 没训好，训练就极端化了
                        train_pred_extreme_poor_cnt += 1
                        train_pred_extreme_poor_dict[token] = stat




        #if token in vocab_zh_list: # 把unk排除
        if token in vocab_zh_list and stat['train_freq'] >= num_threshold:    # 极端化的num_threshold
            test_pred_ok_ratio_list.append(test_pred_ok_ratio)
            test_gold_ok_ratio_list.append(test_gold_ok_ratio)
            train_pred_ok_ratio_list.append(train_pred_ok_ratio)
            train_gold_ok_ratio_list.append(train_gold_ok_ratio)
        # 挑token：【训练集全标bad】而且【在拟合非常好的情况下测试全预测ok】
        if train_gold_ok_ratio == 0 and test_pred_ok_ratio == 1:
            train_gold_ok_0_test_pred_ok_1.append(token)
        elif train_gold_ok_ratio == 1 and test_pred_ok_ratio == 0:
            train_gold_ok_1_test_pred_ok_0.append(token)
        # 倾向
        if train_gold_ok_ratio != 0.5:
            test_pred_ok_ratio_tendency = round(test_pred_ok_ratio)
            train_pred_ok_ratio_tendency = round(train_pred_ok_ratio)
            train_gold_ok_ratio_tendency = round(train_gold_ok_ratio)
            test_pred_ok_ratio_list_tendency.append(test_pred_ok_ratio_tendency)
            train_pred_ok_ratio_list_tendency.append(train_pred_ok_ratio_tendency)
            train_gold_ok_ratio_list_tendency.append(train_gold_ok_ratio_tendency)
            # 极端化
            if test_pred_ok_ratio_tendency == train_gold_ok_ratio_tendency and \
                    abs(test_pred_ok_ratio_tendency - test_pred_ok_ratio) < abs(train_gold_ok_ratio_tendency - train_gold_ok_ratio):
                extreme_cnt += 1
            if test_pred_ok_ratio_tendency == train_gold_ok_ratio_tendency and \
                    abs(test_pred_ok_ratio_tendency - test_pred_ok_ratio) == abs(train_gold_ok_ratio_tendency - train_gold_ok_ratio):
                same_cnt += 1
            if test_pred_ok_ratio_tendency != train_gold_ok_ratio_tendency or \
                    abs(test_pred_ok_ratio_tendency - test_pred_ok_ratio) > abs(train_gold_ok_ratio_tendency - train_gold_ok_ratio):
                anti_extreme_cnt += 1
            
            all_cnt += 1
    extreme_stat_dict = sorted(extreme_stat_dict.items(), key = lambda x : x[1]['test_freq'], reverse = True)
    with open(extreme_stat_path, 'w', encoding='utf-8') as fs:
        fs.write((json.dumps(extreme_stat_dict, indent = 1)).encode('utf-8').decode("unicode-escape"))
        #json.dump(extreme_stat_dict, fs, indent=1)
    train_pred_extreme_poor_dict = sorted(train_pred_extreme_poor_dict.items(), key = lambda x : x[1]['train_freq'], reverse = True)
"""
print(token_cnt_wo_unk)
print(freq_cnt)
print(extreme_cnt_1)
print(extreme_cnt_2)
print(extreme_cnt_3)
print(extreme_cnt_23)
print(anti_extreme)
"""
print(token_cnt_wo_unk)
print(train_pred_extreme_cnt) # train_pred完全极端化
print(train_freq_cnt)  # 完全极端化中，有多少是超过阈值的
print(train_pred_extreme_poor_cnt)  # train_gold不极端

for token, stat in train_pred_extreme_poor_dict:
    print(token + '\t' + str(stat['train_freq']))
assert 1==2
for token, stat in extreme_stat_dict:
    print(token + '\t' + str(stat['test_freq']))


def print_correlation():
    print("============= 标签比例相关性")
    print(len(test_pred_ok_ratio_list))
    print('pearsonr(test_pred_ok_ratio, train_gold_ok_ratio)')
    print(pearsonr(test_pred_ok_ratio_list, train_gold_ok_ratio_list))
    print('pearsonr(test_pred_ok_ratio, train_pred_ok_ratio)')
    print(pearsonr(test_pred_ok_ratio_list, train_pred_ok_ratio_list))

    print("============= 标签倾向相关性")
    print(len(test_pred_ok_ratio_list_tendency))
    print('pearsonr(test_pred_ok_ratio_tendency, train_gold_ok_ratio_tendency)')
    print(pearsonr(test_pred_ok_ratio_list_tendency, train_gold_ok_ratio_list_tendency))
    print('pearsonr(test_pred_ok_ratio_tendency, train_pred_ok_ratio_tendency)')
    print(pearsonr(test_pred_ok_ratio_list_tendency, train_pred_ok_ratio_list_tendency))

    print("============= 极端化标签占比")  # 基线 50%
    print(extreme_cnt)
    print(same_cnt)
    print(anti_extreme_cnt)
    print(all_cnt)
    print(extreme_cnt / all_cnt)
    print(same_cnt / all_cnt)
    print(anti_extreme_cnt / all_cnt)

#print_correlation()

# 统计
"""
num_gold_extreme = 0
num_pred_same_extreme = 0
for r1, r2 in zip(sorted_train_gold_ok_ratio, sorted_test_pred_ok_ratio):
    if r1 == 0 or r1 == 1:
        num_gold_extreme += 1
        if r1 == r2:
            num_pred_same_extreme += 1

print("----------------------")
print(num_gold_extreme)
print(num_pred_same_extreme)
print(num_pred_same_extreme / num_gold_extreme)
assert 1==2
"""

# 挑token：【训练集全标bad】而且【在拟合非常好的情况下测试全预测ok】
"""
print("train_gold_ok_0_test_pred_ok_1")
print(len(train_gold_ok_0_test_pred_ok_1))
print(train_gold_ok_0_test_pred_ok_1)
print("train_gold_ok_1_test_pred_ok_0")
print(len(train_gold_ok_1_test_pred_ok_0))
print(train_gold_ok_1_test_pred_ok_0)
train_gold_ok_0_test_pred_ok_1_path = "/home/yanym/save/enzh/anti_extreme/train_gold_ok_0_test_pred_ok_1"
train_gold_ok_1_test_pred_ok_0_path = "/home/yanym/save/enzh/anti_extreme/train_gold_ok_1_test_pred_ok_0"
with open(train_gold_ok_0_test_pred_ok_1_path, 'w', encoding='utf-8') as f1:
    for token in train_gold_ok_0_test_pred_ok_1:
        f1.write(token + '\n')
with open(train_gold_ok_1_test_pred_ok_0_path, 'w', encoding='utf-8') as f1:
    for token in train_gold_ok_1_test_pred_ok_0:
        f1.write(token + '\n')

assert 1==2
"""



# python scripts/analysis/statistics/completely_extreme.py