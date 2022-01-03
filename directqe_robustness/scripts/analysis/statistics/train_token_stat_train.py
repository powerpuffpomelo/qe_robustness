"""
对每一个 test和qe_train交集的token：
- 模型预测结果okbad比例 和 训练时gold标签okbad比例 的相关性
- 模型预测结果okbad比例 和 模型在训练集上pred标签okbad比例 的相关性
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
#qe_test_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/test_result/test.tag"
qe_test_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_ckpt_end_end/test_result/test.tag"
qe_test_gold_tag_path = "/home/yanym/save/enzh/vocab/test.tag_gold"

qe_train_mt_path = "/home/yanym/save/enzh/vocab/train.mt"
#qe_train_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_ckpt/test_result_ckpt/train.tag.22800"
#qe_train_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/test_result/train.tag"
qe_train_pred_tag_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_ckpt_end_end/test_result/train.tag"
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

with open(save_path, 'r', encoding='utf-8') as f_stat:
    token_stat = json.load(f_stat)
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
        #if token in vocab_zh_list: # 把unk排除
        if token in vocab_zh_list and stat['train_freq'] >= num_threshold:       # 极端化的num_threshold
            test_pred_ok_ratio_list.append(test_pred_ok_ratio)
            test_gold_ok_ratio_list.append(test_gold_ok_ratio)
            train_pred_ok_ratio_list.append(train_pred_ok_ratio)
            train_gold_ok_ratio_list.append(train_gold_ok_ratio)
        # 挑token：【训练集全标bad】而且【在拟合非常好的情况下测试全预测ok】
        if train_gold_ok_ratio == 0 and test_pred_ok_ratio == 1:
            train_gold_ok_0_test_pred_ok_1.append(token)
        elif train_gold_ok_ratio == 1 and test_pred_ok_ratio == 0:
            train_gold_ok_1_test_pred_ok_0.append(token)
        # 画图备用
        """
        plt_test_pred.append(test_pred_ok_ratio)
        plt_train_gold.append(train_gold_ok_ratio)
        if math.floor(test_pred_ok_ratio * 10) in plt_test_pred_dict.keys():
            plt_test_pred_dict[math.floor(test_pred_ok_ratio * 10)] += 1
        else:
            plt_test_pred_dict[math.floor(test_pred_ok_ratio * 10)] = 1
        if math.floor(train_gold_ok_ratio * 10) in plt_train_gold_dict.keys():
            plt_train_gold_dict[math.floor(train_gold_ok_ratio * 10)] += 1
        else:
            plt_train_gold_dict[math.floor(train_gold_ok_ratio * 10)] = 1
        """
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

# 画图1，并列柱状图

def plot_bar():
    plt_test_pred_dict[9] += plt_test_pred_dict[10]
    del plt_test_pred_dict[10]
    plt_test_pred_dict = dict(sorted(plt_test_pred_dict.items(), key = lambda d : d[0]))

    plt_train_gold_dict[9] += plt_train_gold_dict[10]
    del plt_train_gold_dict[10]
    plt_train_gold_dict = dict(sorted(plt_train_gold_dict.items(), key = lambda d : d[0]))

    x = np.arange(10)
    tick_label = []

    bar_width = 0.3 #设置柱状图的宽度
    tick_list = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
    for i in range(10):
        tick_label.append(tick_list[i] + '-' + tick_list[i+1])

    f, ax = plt.subplots(figsize = (7,7))
    plt.bar(x, plt_train_gold_dict.values(), bar_width, color='c', label='train_gold_tag', alpha=0.5)
    plt.bar(x + bar_width, plt_test_pred_dict.values(), bar_width, color='b', label='test_pred_tag', alpha=0.5)

    plt.legend()#显示图例，即label
    plt.xticks(x + bar_width / 2, tick_label, rotation=30)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置

    #plt.hist(plt_train_gold, bins=10, facecolor='g',alpha=0.75)
    plt.title('Extremity of labeling',fontsize=18)
    plt.xlabel('ok ratio', fontsize=16)
    plt.ylabel('num', fontsize=16)
    #plt.grid(True)

    plt.savefig('/home/yanym/save/enzh/stat/hist_test.png')

#plot_bar()


# 画图2，极端化图
train_test_ok_zip = list(zip(train_gold_ok_ratio_list, train_pred_ok_ratio_list, test_gold_ok_ratio_list, test_pred_ok_ratio_list))
#print(train_test_ok_zip)
train_test_ok_zip = sorted(train_test_ok_zip, key=lambda x:x[0])
#print(train_test_ok_zip)
sorted_train_gold_ok_ratio, sorted_train_pred_ok_ratio, sorted_test_gold_ok_ratio, sorted_test_pred_ok_ratio = list(zip(*train_test_ok_zip))

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

"""
y1 = np.array(sorted_train_gold_ok_ratio)
y2 = np.array(sorted_test_gold_ok_ratio)
np.save('/home/yanym/save/enzh/stat/y1.npy', y1)
np.save('/home/yanym/save/enzh/stat/y2.npy', y2)
assert 1==2
"""
x = np.arange(len(sorted_train_gold_ok_ratio))
fig, ax = plt.subplots(figsize=(40,15))
ax.scatter(x, sorted_train_gold_ok_ratio, label="train_gold_tag", marker='x', color='gold', alpha=0.5)
ax.scatter(x, sorted_train_pred_ok_ratio, label="train_pred_tag", marker='x', color='aquamarine', alpha=1)
ax.scatter(x, sorted_test_gold_ok_ratio, label="test_gold_tag", marker='x', color='tomato', alpha=1)
ax.scatter(x, sorted_test_pred_ok_ratio, label="test_pred_tag", marker='x', color='deepskyblue', alpha=1)
ax.set_xlabel('every token', fontsize=24)
ax.set_ylabel('ok ratio', fontsize=24)
ax.set_title("Extremity of labeling", fontsize=28)
ax.legend(fontsize=24)
plt.savefig('/home/yanym/save/enzh/stat/extreme_overfitting_four.png')

x = np.arange(len(sorted_train_gold_ok_ratio))
fig, ax = plt.subplots(figsize=(40,15))
ax.scatter(x, sorted_train_gold_ok_ratio, label="train_gold_tag", marker='x', color='gold', alpha=1)
ax.set_xlabel('every token', fontsize=24)
ax.set_ylabel('ok ratio', fontsize=24)
ax.set_title("Extremity of labeling", fontsize=28)
ax.legend(fontsize=24)
plt.savefig('/home/yanym/save/enzh/stat/extreme_overfitting_1.png')

x = np.arange(len(sorted_train_gold_ok_ratio))
fig, ax = plt.subplots(figsize=(40,15))
ax.scatter(x, sorted_train_pred_ok_ratio, label="train_pred_tag", marker='x', color='aquamarine', alpha=1)
ax.set_xlabel('every token', fontsize=24)
ax.set_ylabel('ok ratio', fontsize=24)
ax.set_title("Extremity of labeling", fontsize=28)
ax.legend(fontsize=24)
plt.savefig('/home/yanym/save/enzh/stat/extreme_overfitting_2.png')

x = np.arange(len(sorted_train_gold_ok_ratio))
fig, ax = plt.subplots(figsize=(40,15))
ax.scatter(x, sorted_test_gold_ok_ratio, label="test_gold_tag", marker='x', color='tomato', alpha=1)
ax.set_xlabel('every token', fontsize=24)
ax.set_ylabel('ok ratio', fontsize=24)
ax.set_title("Extremity of labeling", fontsize=28)
ax.legend(fontsize=24)
plt.savefig('/home/yanym/save/enzh/stat/extreme_overfitting_3.png')

x = np.arange(len(sorted_train_gold_ok_ratio))
fig, ax = plt.subplots(figsize=(40,15))
ax.scatter(x, sorted_test_pred_ok_ratio, label="test_pred_tag", marker='x', color='deepskyblue', alpha=1)
ax.set_xlabel('every token', fontsize=24)
ax.set_ylabel('ok ratio', fontsize=24)
ax.set_title("Extremity of labeling", fontsize=28)
ax.legend(fontsize=24)
plt.savefig('/home/yanym/save/enzh/stat/extreme_overfitting_4.png')


# python scripts/analysis/statistics/train_token_stat_train.py