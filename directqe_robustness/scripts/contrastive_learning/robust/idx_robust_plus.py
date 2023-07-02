# 之前的robust idx 只标明了 原始+增强 中 改变的部分
# 这里把原始数据的idx变成所有，增强数据中idx依然是改变的部分
"""
file_mt = "/data1/yanym/data/robust_language_model_wmt21/train_limit3_10_double/robust_train.mt"
file_idx_ori = "/data1/yanym/data/robust_language_model_wmt21/train_limit3_10_double/robust_train.idx"
file_idx_plus = "/data1/yanym/data/robust_language_model_wmt21/train_limit3_10_double/robust_train.idx_plus"

with open(file_mt, 'r', encoding='utf-8') as fmt, open(file_idx_ori, 'r', encoding='utf-8') as fori, \
    open(file_idx_plus, 'w', encoding='utf-8') as fplus:
    mt_lines = fmt.readlines()
    idx_ori_lines = fori.readlines()
    i = 0
    while i < len(mt_lines):
        print(i)
        if i == 0 or mt_lines[i] != mt_lines[i - 1]:  # 开启一个新的样本
            fplus.write(" ".join(map(str, list(range(len(mt_lines[i].strip('\n').split()))))) + "\n")
        else:
            fplus.write(idx_ori_lines[i])
        i += 1
"""

# plus2origin
# 仅限训练记忆
import json
train_tag_stat_path = "/data1/yanym/data/robust_train_memory/train_token_tag_stat.json"
with open(train_tag_stat_path, 'r', encoding='utf-8') as f:
    train_tag_stat = json.load(f)

file_mt = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update/robust_train.mt"
file_mt_tag = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update/robust_train.mt_tag"
file_idx_plus = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update/robust_train.mt_idx_plus"
file_idx_contrast = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update/robust_train.mt_idx_contrast"
file_idx_popular = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update/robust_train.mt_idx_popular"

with open(file_mt, 'r', encoding='utf-8') as fmt, open(file_mt_tag, 'r', encoding='utf-8') as ftag, \
    open(file_idx_plus, 'r', encoding='utf-8') as fplus, \
    open(file_idx_contrast, 'w', encoding='utf-8') as fcontrast, \
    open(file_idx_popular, 'w', encoding='utf-8') as fpopular:
    mt_lines = fmt.readlines()
    tag_lines = ftag.readlines()
    idx_plus_lines = fplus.readlines()
    i = 0
    cnt = 0
    while i < len(mt_lines):
        print(i)
        if cnt == 0:  # 开启一个新的样本
            token_id = 0
            sent = mt_lines[i].strip('\n').split()
            tag_line = tag_lines[i].strip('\n').split()
            for token, tag in zip(sent, tag_line):
                if ((tag == 'OK' and train_tag_stat[token]['ok_ratio'] > 0.5) or (tag == 'BAD' and train_tag_stat[token]['ok_ratio'] < 0.5 and train_tag_stat[token]['ok_freq'] != 0)):
                    fcontrast.write(str(token_id) + ' ')
                    fpopular.write(str(token_id) + ' ')
                    cnt += 1
                token_id += 1
            fcontrast.write('\n')
            fpopular.write('\n')
        else:
            fcontrast.write(idx_plus_lines[i])
            fpopular.write('\n')
            cnt -= 1
        i += 1


# python scripts/contrastive_learning/robust/idx_robust_plus.py