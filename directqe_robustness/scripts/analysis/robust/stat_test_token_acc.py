# 模型对某一集合的预测tag，存下来其中出现过的每个token及其准确率
# 键：token，值：正确次数，错误次数，acc

import json

mt_bpe_path = "/data1/yanym/data/robust_qe_train_word2vec_double/robust_train.mt.BPE"
robust_idx_path = "/data1/yanym/data/robust_qe_train_word2vec_double/robust_train.idx.BPE"
gold_tag_path = "/data1/yanym/data/robust_qe_train_word2vec_double/robust_train.tag_idxonly.BPE"
pred_tag_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/test_result_9600/robust_train_word2vec_double.tag_idxonly.BPE" 
stat_save_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/test_result_9600/robust_train_word2vec_double_stat.json"

stat_dict = dict()
with open(mt_bpe_path, 'r', encoding='utf-8') as fmt, \
    open(robust_idx_path, 'r', encoding='utf-8') as fidx, \
    open(gold_tag_path, 'r', encoding='utf-8') as fgold, \
    open(pred_tag_path, 'r', encoding='utf-8') as fpred:
    mt_lines = fmt.readlines()
    idx_lines = fidx.readlines()
    gold_tag_lines = fgold.readlines()
    pred_tag_lines = fpred.readlines()
    for mt_line, idx_line, gold_tag_line, pred_tag_line in zip(mt_lines, idx_lines, gold_tag_lines, pred_tag_lines):
        
        mt_line = mt_line.strip('\n').split()
        idx_line = idx_line.strip('\n').split()
        gold_tag_line = gold_tag_line.strip('\n').split()
        pred_tag_line = pred_tag_line.strip('\n').split()

        mt_line_only_idx = []
        for idx in idx_line:
            mt_line_only_idx.append(mt_line[int(idx)])

        for mt_token, gold_tag, pred_tag in zip(mt_line_only_idx, gold_tag_line, pred_tag_line):
            if mt_token not in stat_dict:
                stat_dict[mt_token] = {'freq':0, 'right_freq':0, 'wrong_freq':0, 'acc':0}
            stat_dict[mt_token]['freq'] += 1
            if gold_tag == pred_tag: stat_dict[mt_token]['right_freq'] += 1
            else: stat_dict[mt_token]['wrong_freq'] += 1

for token in stat_dict:
    stat_dict[token]['acc'] = stat_dict[token]['right_freq'] / stat_dict[token]['freq']

with open(stat_save_path, 'w', encoding='utf-8') as f_s:
    json.dump(stat_dict, f_s, indent = 1, ensure_ascii=False)

# python scripts/analysis/robust/stat_test_token_acc.py