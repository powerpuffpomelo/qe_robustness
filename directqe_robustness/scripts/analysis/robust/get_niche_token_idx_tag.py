# 根据训练时大部分被打什么标签，得到测试集or训练集的小众tag文件，和mt对应，得到01文件，0代表大众或持平，1代表小众, -1 代表没在训练集出现过
import json

train_tag_stat_path = "/data1/yanym/data/robust_train_memory/train_token_tag_stat.json"
qe_test_mt_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt"
qe_test_gold_tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag"
qe_test_niche_tag_path = "/data1/yanym/data/robust_train_memory/test.niche_tag"

with open(train_tag_stat_path, 'r', encoding='utf-8') as f:
    train_tag_stat = json.load(f)

with open(qe_test_mt_path, 'r', encoding='utf-8') as f_mt, open(qe_test_gold_tag_path, 'r', encoding='utf-8') as f_tag, \
    open(qe_test_niche_tag_path, 'w', encoding='utf-8') as f_save:
    for line_mt, line_tag in zip(f_mt.readlines(), f_tag.readlines()):
        line_mt = line_mt.strip('\n').split()
        line_tag = line_tag.strip('\n').split()
        line_niche = []
        for token, tag in zip(line_mt, line_tag):
            if token in train_tag_stat:
                if tag == 'OK' and train_tag_stat[token]['ok_ratio'] >= 0.5: line_niche.append(0)
                elif tag == 'BAD' and train_tag_stat[token]['ok_ratio'] <= 0.5: line_niche.append(0)
                else: line_niche.append(1)
            else:
                line_niche.append(-1)
        f_save.write(' '.join(map(str, line_niche)) + '\n')

# python scripts/analysis/robust/get_niche_token_idx_tag.py