# 找到dev集对应的pe
"""
need_dev_src = '/home/yanym/save/enzh/generate_dev/dev.src'
all_dev_src = '/home/data_ti4_c/yanym/data/wmt20/enzh/task2/dev/dev.src'
all_dev_pe = '/home/data_ti4_c/yanym/data/wmt20/enzh/task2/dev/dev.pe'
need_dev_pe = '/home/yanym/save/enzh/generate_dev/dev.pe'

with open(need_dev_src, 'r', encoding = 'utf-8') as f_need_src, \
    open(all_dev_src, 'r', encoding = 'utf-8') as f_all_src, \
    open(all_dev_pe, 'r', encoding = 'utf-8') as f_all_pe, \
    open(need_dev_pe, 'w', encoding = 'utf-8') as f_need_pe:
    need_dev_lines = []
    need_src_lines = f_need_src.readlines()
    all_src_lines = f_all_src.readlines()
    all_pe_lines = f_all_pe.readlines()
    for need_src_line in need_src_lines:
        for all_src_line, all_pe_line in zip(all_src_lines, all_pe_lines):
            if all_src_line == need_src_line:
                need_dev_lines.append(all_pe_line)
                break
    f_need_pe.writelines(need_dev_lines)
"""

# 从7000中取3500个数，抽出伪数据，另外3500抽出真实数据
import numpy as np
"""
true_prefix = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/"
pseudo_prefix = "/home/yanym/save/enzh/generate_dev/"
save_prefix = "/home/yanym/save/enzh/generate_dev/"

sample_num = 3500
sample_ids = np.random.choice(range(7000), size=sample_num, replace=False)

with open(true_prefix + "train.src.BPE", 'r', encoding='utf-8') as f1,\
    open(true_prefix + "train.mt", 'r', encoding='utf-8') as f2,\
    open(true_prefix + "train.tag", 'r', encoding='utf-8') as f3,\
    open(true_prefix + "train.hter", 'r', encoding='utf-8') as f4,\
    open(save_prefix + "qe_train_3500.src.BPE", 'w', encoding='utf-8') as f5,\
    open(save_prefix + "qe_train_3500.mt", 'w', encoding='utf-8') as f6,\
    open(save_prefix + "qe_train_3500.tag", 'w', encoding='utf-8') as f7,\
    open(save_prefix + "qe_train_3500.hter", 'w', encoding='utf-8') as f8:
    src_all = f1.readlines()
    mt_all = f2.readlines()
    tag_all = f3.readlines()
    hter_all = f4.readlines()
    zip_data = list(zip(src_all, mt_all, tag_all, hter_all))
    for i in range(7000):
        if i not in sample_ids:
            sample_src, sample_mt, sample_tag, sample_hter = zip_data[i]
            f5.writelines(sample_src)
            f6.writelines(sample_mt)
            f7.writelines(sample_tag)
            f8.writelines(sample_hter)

with open(pseudo_prefix + "qe_train_pseudo_7000.src.BPE", 'r', encoding='utf-8') as f1,\
    open(pseudo_prefix + "qe_train_pseudo_7000.mt", 'r', encoding='utf-8') as f2,\
    open(pseudo_prefix + "qe_train_pseudo_7000.tag", 'r', encoding='utf-8') as f3,\
    open(pseudo_prefix + "qe_train_pseudo_7000.hter", 'r', encoding='utf-8') as f4,\
    open(save_prefix + "qe_train_pseudo_3500.src.BPE", 'w', encoding='utf-8') as f5,\
    open(save_prefix + "qe_train_pseudo_3500.mt", 'w', encoding='utf-8') as f6,\
    open(save_prefix + "qe_train_pseudo_3500.tag", 'w', encoding='utf-8') as f7,\
    open(save_prefix + "qe_train_pseudo_3500.hter", 'w', encoding='utf-8') as f8:
    src_all = f1.readlines()
    mt_all = f2.readlines()
    tag_all = f3.readlines()
    hter_all = f4.readlines()
    zip_data = list(zip(src_all, mt_all, tag_all, hter_all))
    for i in range(7000):
        if i in sample_ids:
            sample_src, sample_mt, sample_tag, sample_hter = zip_data[i]
            f5.writelines(sample_src)
            f6.writelines(sample_mt)
            f7.writelines(sample_tag)
            f8.writelines(sample_hter)
"""

# 分别得到两批toy train
sample_num = 500
sample_ids = np.random.choice(range(3500), size=sample_num, replace=False)
all_prefix = "/home/yanym/save/enzh/generate_dev/"
save_prefix = "/home/yanym/save/enzh/generate_dev/"

with open(all_prefix + "qe_train_pseudo_3500.src.BPE", 'r', encoding='utf-8') as f1,\
    open(all_prefix + "qe_train_pseudo_3500.mt", 'r', encoding='utf-8') as f2,\
    open(all_prefix + "qe_train_pseudo_3500.tag", 'r', encoding='utf-8') as f3,\
    open(all_prefix + "qe_train_pseudo_3500.hter", 'r', encoding='utf-8') as f4,\
    open(save_prefix + "qe_train_pseudo_toy.src.BPE", 'w', encoding='utf-8') as f5,\
    open(save_prefix + "qe_train_pseudo_toy.mt", 'w', encoding='utf-8') as f6,\
    open(save_prefix + "qe_train_pseudo_toy.tag", 'w', encoding='utf-8') as f7,\
    open(save_prefix + "qe_train_pseudo_toy.hter", 'w', encoding='utf-8') as f8:
    src_all = f1.readlines()
    mt_all = f2.readlines()
    tag_all = f3.readlines()
    hter_all = f4.readlines()
    zip_data = list(zip(src_all, mt_all, tag_all, hter_all))
    for i in sample_ids:
        sample_src, sample_mt, sample_tag, sample_hter = zip_data[i]
        f5.writelines(sample_src)
        f6.writelines(sample_mt)
        f7.writelines(sample_tag)
        f8.writelines(sample_hter)

with open(all_prefix + "qe_train_3500.src.BPE", 'r', encoding='utf-8') as f1,\
    open(all_prefix + "qe_train_3500.mt", 'r', encoding='utf-8') as f2,\
    open(all_prefix + "qe_train_3500.tag", 'r', encoding='utf-8') as f3,\
    open(all_prefix + "qe_train_3500.hter", 'r', encoding='utf-8') as f4,\
    open(save_prefix + "qe_train_toy.src.BPE", 'w', encoding='utf-8') as f5,\
    open(save_prefix + "qe_train_toy.mt", 'w', encoding='utf-8') as f6,\
    open(save_prefix + "qe_train_toy.tag", 'w', encoding='utf-8') as f7,\
    open(save_prefix + "qe_train_toy.hter", 'w', encoding='utf-8') as f8:
    src_all = f1.readlines()
    mt_all = f2.readlines()
    tag_all = f3.readlines()
    hter_all = f4.readlines()
    zip_data = list(zip(src_all, mt_all, tag_all, hter_all))
    for i in sample_ids:
        sample_src, sample_mt, sample_tag, sample_hter = zip_data[i]
        f5.writelines(sample_src)
        f6.writelines(sample_mt)
        f7.writelines(sample_tag)
        f8.writelines(sample_hter)

# python scripts/analysis/statistics/qe_pseudo_process.py