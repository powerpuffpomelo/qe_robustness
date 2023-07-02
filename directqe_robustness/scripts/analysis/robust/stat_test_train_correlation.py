# 模型在鲁棒性测试集和鲁棒性训练集中，预测准确率的相关性

import json
from scipy.stats import pearsonr

test_stat_dict_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/test_result_9600/robust_test_word2vec_limit_3_10_stat.json"
train_stat_dict_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/test_result_9600/robust_train_word2vec_double_stat.json"

def pearson_corrcoef(x, y):
    return pearsonr(x, y)[0]

test_acc_list = []
train_acc_list = []
train_freq_list = []
with open(test_stat_dict_path, 'r', encoding='utf-8') as ftest, \
    open(train_stat_dict_path, 'r', encoding='utf-8') as ftrain:
    test_stat = json.load(ftest)
    train_stat = json.load(ftrain)

    for token, stat in test_stat.items():
        if token in train_stat:
            test_acc_list.append(stat['acc'])
            train_acc_list.append(train_stat[token]['acc'])
            train_freq_list.append(train_stat[token]['freq'])


print ("test_train_acc  pearson_corrcoef = %.4f" % pearson_corrcoef(test_acc_list, train_acc_list))
print ("test_train_freq  pearson_corrcoef = %.4f" % pearson_corrcoef(test_acc_list, train_freq_list))

# python scripts/analysis/robust/stat_test_train_correlation.py