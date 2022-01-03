# 统计【在伪数据/qe训练集 出现/没出现】的token 在测试集上的效果
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

"""
qe_test_mt_path = "/home/yanym/save/enzh/vocab2/test.mt"
qe_test_pred_tag_path = "/home/yanym/save/enzh/vocab2/test.tag_qe_fifi"
qe_test_gold_tag_path = "/home/yanym/save/enzh/vocab2/test.tag_gold"
pseudo_vocab_path = "/home/yanym/save/enzh/vocab2/qe_train_pseudo_3500.mt.json"
qe_train_vocab_path = "/home/yanym/save/enzh/vocab2/qe_train_3500.mt.json"
save_path = "/home/yanym/save/enzh/vocab2/qe_test_token_stat.json"
"""
qe_test_mt_path = "/home/yanym/save/enzh/vocab/train.mt"
qe_test_pred_tag_path = "/home/yanym/save/enzh/vocab/train.tag_fi"
qe_test_gold_tag_path = "/home/yanym/save/enzh/vocab/train.tag_gold"
pseudo_vocab_path = "/home/yanym/save/enzh/vocab/pseudo_train_v2.mt.json"
qe_train_vocab_path = "/home/yanym/save/enzh/vocab/qe_train.mt.json"
save_path = "/home/yanym/save/enzh/vocab/qe_test_token_stat.json"

# 构造token大字典
def make_stat_dict(file_mt, file_pred, file_gold, pseudo_vocab_path, qe_train_vocab_path, save_path):
    ret_dict = dict()
    with open(file_mt, 'r', encoding='utf-8') as f_mt, open(file_pred, 'r', encoding='utf-8') as f_pred, \
        open(file_gold, 'r', encoding='utf-8') as f_gold:
        for line_mt, line_pred, line_gold in zip(f_mt.readlines(), f_pred.readlines(), f_gold.readlines()):
            line_mt = line_mt.strip('\n').split()
            line_pred = line_pred.strip('\n').split()
            line_gold = line_gold.strip('\n').split()
            for i, token in enumerate(line_mt):
                if token not in ret_dict:
                    ret_dict[token] = {'freq':1, 'in_pseudo':False, 'in_qe_train':False, 'tag_pred':[], 'tag_gold':[], 'acc':0}
                else:
                    ret_dict[token]['freq'] += 1
                ret_dict[token]['tag_pred'].append(line_pred[i])
                ret_dict[token]['tag_gold'].append(line_gold[i])
    with open(pseudo_vocab_path, 'r', encoding='utf-8') as f_p, open(qe_train_vocab_path, 'r', encoding='utf-8') as f_q:
        pseudo_vocab = json.load(f_p)
        qe_train_vocab = json.load(f_q)
    for token, stat in ret_dict.items():
        if token in pseudo_vocab:
            stat['in_pseudo'] = True
        if token in qe_train_vocab:
            stat['in_qe_train'] = True
        stat['acc'] = accuracy_score(stat['tag_pred'], stat['tag_gold'])
    with open(save_path, 'w', encoding='utf-8') as f_s:
        json.dump(ret_dict, f_s, indent=1)

#"""
make_stat_dict(file_mt=qe_test_mt_path,
                file_pred=qe_test_pred_tag_path,
                file_gold=qe_test_gold_tag_path,
                pseudo_vocab_path=pseudo_vocab_path,
                qe_train_vocab_path=qe_train_vocab_path,
                save_path=save_path)
#"""

preList = []
goldList = []
sum_acc = 0
num_token = 0

with open(save_path, 'r', encoding='utf-8') as f_stat:
    token_stat = json.load(f_stat)
    for token, stat in token_stat.items():
        if stat['in_pseudo'] == False and stat['in_qe_train']:
            preList.extend(stat['tag_pred'])
            goldList.extend(stat['tag_gold'])
            sum_acc += stat['acc']
            num_token += 1
    preList = np.array(preList)
    goldList = np.array(goldList)
    macro_acc = sum_acc / num_token

    int_prelist = [1 if token == 'OK' else 0 for token in preList]
    int_goldlist = [1 if token == 'OK' else 0 for token in goldList]
    
    pred_num_ok = sum(int_prelist)
    pred_num_bad = len(int_prelist) - pred_num_ok
    gold_num_ok = sum(int_goldlist)
    gold_num_bad = len(int_goldlist) - gold_num_ok

    acc = accuracy_score(goldList, preList)
    #print(f1_score(goldList, preList, average=None, pos_label=None))
    #print(f1_score(goldList, preList, average=None, pos_label=None).dtype)
    f1_bad, f1_ok = f1_score(goldList, preList, average=None, pos_label=None)
    precision_bad, precision_ok = precision_score(goldList, preList, average=None)
    recall_bad, recall_ok = recall_score(goldList, preList, average=None)
    print("num_token = %d" % len(preList))
    print("num_token_type = %d" % num_token)
    print("micro acc = %.4f" % acc)
    print("macro acc = %.4f" % macro_acc)
    print("f1-ok = %.4f" % f1_ok)
    print("f1-bad = %.4f" % f1_bad)
    print("f1-mult = %.4f" % (f1_bad*f1_ok))

    print("precision_ok = %.4f" % precision_ok)
    print("precision_bad = %.4f" % precision_bad)
    print("recall_ok = %.4f" % recall_ok)
    print("recall_bad = %.4f" % recall_bad)

    print("gold_num_ok = %d" % gold_num_ok)
    print("pred_num_ok = %d" % pred_num_ok)
    print("gold_num_bad = %d" % gold_num_bad)
    print("pred_num_bad = %d" % pred_num_bad)

def part_stat(all_path, part_path, in_pseudo, in_qe_train):
    part_stat = dict()
    with open(all_path, 'r', encoding='utf-8') as f_stat:
        token_stat = json.load(f_stat)
        for token, stat in token_stat.items():
            if stat['in_pseudo'] == in_pseudo and stat['in_qe_train'] == in_qe_train:
                part_stat[token] = stat
    with open(part_path, 'w', encoding='utf-8') as f_save:
        #json.dump(part_stat, f_save, indent = 1)
        f_save.write((json.dumps(part_stat, indent = 1)).encode('utf-8').decode("unicode-escape"))

part_path = "/home/yanym/save/enzh/vocab/neither_stat.json"
#part_stat(save_path, part_path, False, False)

# python scripts/analysis/statistics/train_token_stat.py