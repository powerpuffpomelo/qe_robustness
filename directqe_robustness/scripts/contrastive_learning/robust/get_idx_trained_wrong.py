# 得到训练集上没训好、预测错误的idx
"""
gold_tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.tag.BPE"
pred_tag_path = "/data1/yanym/save/temp/train.tag.BPE"
save_idx_path = "/data1/yanym/data/robust_boosting/train.trained_wrong_idx.BPE"

with open(gold_tag_path, 'r', encoding='utf-8') as fg, open(pred_tag_path, 'r', encoding='utf-8') as fp, \
    open(save_idx_path, 'w', encoding='utf-8') as fs:
    gold_tag_lines = fg.readlines()
    pred_tag_lines = fp.readlines()
    for gold_tag_line, pred_tag_line in zip(gold_tag_lines, pred_tag_lines):
        gold_tag_line = gold_tag_line.strip('\n').split()
        pred_tag_line = pred_tag_line.strip('\n').split()
        wrong_idx_list = []
        token_id = 0
        for gold_tag, pred_tag in zip(gold_tag_line, pred_tag_line):
            if gold_tag != pred_tag:
                wrong_idx_list.append(token_id)
            token_id += 1
        fs.write(' '.join(map(str, wrong_idx_list)) + '\n')
"""

# 得到训练集上训好、预测对的idx

gold_tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.tag.BPE"
pred_tag_path = "/data1/yanym/save/temp/train.tag.BPE"
save_idx_path = "/data1/yanym/data/robust_boosting/train.trained_right_idx.BPE"

with open(gold_tag_path, 'r', encoding='utf-8') as fg, open(pred_tag_path, 'r', encoding='utf-8') as fp, \
    open(save_idx_path, 'w', encoding='utf-8') as fs:
    gold_tag_lines = fg.readlines()
    pred_tag_lines = fp.readlines()
    for gold_tag_line, pred_tag_line in zip(gold_tag_lines, pred_tag_lines):
        gold_tag_line = gold_tag_line.strip('\n').split()
        pred_tag_line = pred_tag_line.strip('\n').split()
        wrong_idx_list = []
        token_id = 0
        for gold_tag, pred_tag in zip(gold_tag_line, pred_tag_line):
            if gold_tag == pred_tag:
                wrong_idx_list.append(token_id)
            token_id += 1
        fs.write(' '.join(map(str, wrong_idx_list)) + '\n')

# python scripts/contrastive_learning/robust/get_idx_trained_wrong.py