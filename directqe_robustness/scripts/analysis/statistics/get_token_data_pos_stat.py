# 得到训练数据词典，每个词是键，维护在数据集中出现的位置（句子id+词id位置）
import json

qe_train_mt_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/train.zh_100w"

save_path = "/data1/yanym/data/robust_train_memory/parallel_word_position_stat.json"

with open(qe_train_mt_path, 'r', encoding='utf-8') as f_mt, \
    open(save_path, 'w', encoding='utf-8') as f_save:
    stat_dict = dict()
    line_id = 0
    for line_mt in f_mt.readlines():
        line_mt = line_mt.strip('\n').split()
        word_id = 0
        for token in line_mt:
            if token not in stat_dict:
                stat_dict[token] = {'appear_freq':0, 'pos_list':[]}
            if stat_dict[token]['appear_freq'] < 3:
                stat_dict[token]['appear_freq'] += 1
                stat_dict[token]['pos_list'].append([line_id, word_id])
            word_id += 1
        line_id += 1
    json.dump(stat_dict, f_save, indent=1, ensure_ascii=False)

# python scripts/analysis/statistics/get_token_data_pos_stat.py