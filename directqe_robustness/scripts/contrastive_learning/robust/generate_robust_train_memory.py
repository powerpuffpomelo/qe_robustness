# 自动生成，基于训练记忆的数据增强鲁棒性训练集
"""
- 增强一批数据，遍历训练集中每个word，如果是大众word的话，就给它一条增强样本，这条增强样本只训练这一个word
  - 如果这个大众word是ok，就从错位文件中把这一条取出来；
  - 如果这个大众word是bad，就从小众样本列表里随机选一个，如果没有就不选
"""
import json
import random

random.seed(1234)

train_tag_stat_path = "/data1/yanym/data/robust_train_memory/train_token_tag_stat.json"
with open(train_tag_stat_path, 'r', encoding='utf-8') as f:
    train_tag_stat = json.load(f)

qe_train_src = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.src"
qe_train_mt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt"
qe_train_tag = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.tag"
qe_train_hter = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.hter"

switch_train_src = "/data1/yanym/data/robust_train_memory/switch_src_mt/switch_train.src"
switch_train_mt = "/data1/yanym/data/robust_train_memory/switch_src_mt/switch_train.mt"
switch_train_tag = "/data1/yanym/data/robust_train_memory/switch_src_mt/switch_train.tag"
switch_train_hter = "/data1/yanym/data/robust_train_memory/switch_src_mt/switch_train.hter"

robust_train_src = "/data1/yanym/data/robust_train_memory/data_augmentation_train/robust_train.src"
robust_train_mt = "/data1/yanym/data/robust_train_memory/data_augmentation_train/robust_train.mt"
robust_train_tag = "/data1/yanym/data/robust_train_memory/data_augmentation_train/robust_train.tag"
robust_train_hter = "/data1/yanym/data/robust_train_memory/data_augmentation_train/robust_train.hter"
robust_train_idx = "/data1/yanym/data/robust_train_memory/data_augmentation_train/robust_train.idx"

with open(qe_train_src, 'r', encoding='utf-8') as f_ori_src, open(qe_train_mt, 'r', encoding='utf-8') as f_ori_mt, \
    open(qe_train_tag, 'r', encoding='utf-8') as f_ori_tag, open(qe_train_hter, 'r', encoding='utf-8') as f_ori_hter, \
    open(switch_train_src, 'r', encoding='utf-8') as f_swi_src, open(switch_train_mt, 'r', encoding='utf-8') as f_swi_mt, \
    open(switch_train_tag, 'r', encoding='utf-8') as f_swi_tag, open(switch_train_hter, 'r', encoding='utf-8') as f_swi_hter, \
    open(robust_train_src, 'w', encoding='utf-8') as f_aug_src, open(robust_train_mt, 'w', encoding='utf-8') as f_aug_mt, \
    open(robust_train_tag, 'w', encoding='utf-8') as f_aug_tag, open(robust_train_hter, 'w', encoding='utf-8') as f_aug_hter, \
    open(robust_train_idx, 'w', encoding='utf-8') as f_aug_idx:
    ori_src_lines = f_ori_src.readlines()
    ori_mt_lines = f_ori_mt.readlines()
    ori_tag_lines = f_ori_tag.readlines()
    ori_hter_lines = f_ori_hter.readlines()
    swi_src_lines = f_swi_src.readlines()
    swi_mt_lines = f_swi_mt.readlines()
    swi_tag_lines = f_swi_tag.readlines()
    swi_hter_lines = f_swi_hter.readlines()
    sent_id = 0

    for ori_src_line, ori_mt_line, ori_tag_line, ori_hter_line in zip(ori_src_lines, ori_mt_lines, ori_tag_lines, ori_hter_lines):
        #if sent_id > 3: break

        # 先抄一遍原始数据
        f_aug_src.write(ori_src_line)
        f_aug_mt.write(ori_mt_line)
        f_aug_tag.write(ori_tag_line)
        f_aug_hter.write(ori_hter_line)

        # 再做数据增强
        ori_mt_line_list = ori_mt_line.strip('\n').split()
        ori_tag_line_list = ori_tag_line.strip('\n').split()
        f_aug_idx.write(' '.join(map(str, list(range(len(ori_mt_line_list))))) + '\n')

        token_id = 0
        for token, tag in zip(ori_mt_line_list, ori_tag_line_list):
            # 遍历每个词，如果一个词是大众词，则分情况讨论，okbad分别采用不同的数据增强方式
            if tag == 'OK' and train_tag_stat[token]['ok_ratio'] > 0.5: 
                f_aug_src.write(swi_src_lines[sent_id])
                f_aug_mt.write(swi_mt_lines[sent_id])
                f_aug_tag.write(swi_tag_lines[sent_id])
                f_aug_hter.write(swi_hter_lines[sent_id])
                f_aug_idx.write(str(token_id) + '\n')
            elif tag == 'BAD' and train_tag_stat[token]['ok_ratio'] < 0.5: 
                if train_tag_stat[token]['ok_freq'] != 0:
                    random_niche = random.choice(train_tag_stat[token]['ok_pos'])
                    f_aug_src.write(ori_src_lines[random_niche[0]])
                    f_aug_mt.write(ori_mt_lines[random_niche[0]])
                    f_aug_tag.write(ori_tag_lines[random_niche[0]])
                    f_aug_hter.write(ori_hter_lines[random_niche[0]])
                    f_aug_idx.write(str(random_niche[1]) + '\n')
            
            token_id += 1

        sent_id += 1
            
# python scripts/contrastive_learning/robust/generate_robust_train_memory.py