# 从test集中挑出 所有token都在训练时见过的test样本，
# 然后比较这个都见过的test子集中的词预测为okbad的比例，和之前训练时被标为okbad的比例是否有关联

import string
import json
import re
from scipy.stats import pearsonr
from sklearn import metrics

data_prefix = "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/"
train_mt_file = "train.mt"
train_tag_gold_file = "train.tag"
test_mt_file = "seen_test/seen_test.mt"
test_tag_pred_file = "seen_test/seen_test_pre.tag"
test_tag_gold_file = "seen_test/seen_test.tag"
json_file = "seen_test/token_pred_stat.json"


token_pred_stat = dict()

# 维护字典，先过一遍test集
with open(data_prefix+test_mt_file, 'r', encoding='utf-8') as f_test_mt, \
    open(data_prefix+test_tag_pred_file, 'r', encoding='utf-8') as f_test_tag_pred, \
    open(data_prefix+test_tag_gold_file, 'r', encoding='utf-8') as f_test_tag_gold:
    #iddd = 0
    for line_mt, line_pred, line_gold in zip(f_test_mt, f_test_tag_pred, f_test_tag_gold):
        line_mt = line_mt.lower()
        tokens = re.split(r"[ ]+", line_mt.strip('\n'))
        line_pred = line_pred.strip('\n').split()
        line_gold = line_gold.strip('\n').split()
        assert len(tokens) == len(line_pred) == len(line_gold)
        #print(tokens)
        for index,token in enumerate(tokens):
            #print(index)
            #print(token)
            if token not in token_pred_stat:
                token_pred_stat[token] = {"train_ok":0, "train_bad":0, "train_ratio":0,
                                        "test_pred_ok":0, "test_pred_bad":0, "test_pred_ratio":0,
                                        "test_gold_ok":0, "test_gold_bad":0, "test_gold_ratio":0}
            #print(line_pred[index])
            #print(line_gold[index])
            if line_pred[index] == "OK":
                token_pred_stat[token]["test_pred_ok"] += 1
            else:
                token_pred_stat[token]["test_pred_bad"] += 1
            if line_gold[index] == "OK":
                token_pred_stat[token]["test_gold_ok"] += 1
            else:
                token_pred_stat[token]["test_gold_bad"] += 1
        #print(token_pred_stat)
        #iddd += 1
        #if iddd % 10 == 0: print(iddd)
        #if iddd == 2: break # 只看一个样本

print('begin train------------------')
# 再过一遍train集，查看刚才test集中token在train集中标注情况
with open(data_prefix+train_mt_file, 'r', encoding='utf-8') as f_train_mt, \
    open(data_prefix+train_tag_gold_file, 'r', encoding='utf-8') as f_train_tag_gold:
    #iddd = 0
    for line_mt, line_gold in zip(f_train_mt, f_train_tag_gold):
        line_mt = line_mt.lower()
        tokens = re.split(r"[ ]+", line_mt.strip('\n'))
        line_gold = line_gold.strip('\n').split()
        assert len(tokens) == len(line_gold)
        #print(tokens)
        for index,token in enumerate(tokens):
            # 只有 train test 交集部分的token才考虑
            if token in token_pred_stat:
                if line_gold[index] == 'OK':
                    token_pred_stat[token]["train_ok"] += 1
                else:
                    token_pred_stat[token]["train_bad"] += 1
        #iddd += 1
        #if iddd % 100 == 0: print(iddd)
        #if iddd == 4: break
#print(token_pred_stat)


# 从统计字典中摘出三个列表：列表长度为包含单词数；每个位置 train标注ok所占比例，test预测ok所占比例，test标注ok所占比例
train_ratio = []
test_pred_ratio = []
test_gold_ratio = []
train_ratio_rounding = []
test_pred_ratio_rounding = []
test_gold_ratio_rounding = []
#epsilon = 1e-7
for token, stat_dd in token_pred_stat.items():
    # 添加到列表
    this_train_ratio = stat_dd["train_ok"] / (stat_dd["train_ok"] + stat_dd["train_bad"])
    this_test_pred_ratio = stat_dd["test_pred_ok"] / (stat_dd["test_pred_ok"] + stat_dd["test_pred_bad"])
    this_test_gold_ratio = stat_dd["test_gold_ok"] / (stat_dd["test_gold_ok"] + stat_dd["test_gold_bad"])

    train_ratio.append(this_train_ratio)
    test_pred_ratio.append(this_test_pred_ratio)
    test_gold_ratio.append(this_test_gold_ratio)

    train_ratio_rounding.append(round(this_train_ratio))
    test_pred_ratio_rounding.append(round(this_test_pred_ratio))
    test_gold_ratio_rounding.append(round(this_test_gold_ratio))

    # 添加到字典
    stat_dd["train_ratio"] = this_train_ratio
    stat_dd["test_pred_ratio"] = this_test_pred_ratio
    stat_dd["test_gold_ratio"] = this_test_gold_ratio

print(len(train_ratio))
#print(train_ratio)
#print(test_pred_ratio)
#print(test_gold_ratio)

# ===================================# 
# 插入小测试
print('qqqqqqqqqqq')
print(token_pred_stat["决赛"])
assert 1==2

# ===================================# 

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

print ("test_pred & train")
cal_info(test_pred_ratio, train_ratio)
print ("test_gold & train")
cal_info(test_gold_ratio, train_ratio)
print ("test_gold & test_pred")
cal_info(test_gold_ratio, test_pred_ratio)
print("--------rounding-----------")
print ("test_pred & train")
cal_info(test_pred_ratio_rounding, train_ratio_rounding)
print ("test_gold & train")
cal_info(test_gold_ratio_rounding, train_ratio_rounding)
print ("test_gold & test_pred")
cal_info(test_gold_ratio_rounding, test_pred_ratio_rounding)

# 写入json文件
with open(data_prefix+json_file, 'w', encoding='utf-8') as f:
    json.dump(token_pred_stat, f, indent=1)


# python scripts/attribution_enzh/seen_token_pred_stat.py