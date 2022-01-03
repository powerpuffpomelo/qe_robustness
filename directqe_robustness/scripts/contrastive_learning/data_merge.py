import os

# 三份finetune数据合并到一起啦
"""
prefix_1 = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/dev/dev."
prefix_2 = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_neg_parallel/dev/dev."
prefix_3 = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_neg_out_of_order/dev/dev."
prefix_save = "/home/yanym/data/qe_merge/dev/dev."

split = ['src', 'src.BPE', 'mt', 'src_tag', 'mt_tag', 'mtgap_tag', 'hter']
for s in split:
    with open(prefix_1 + s, 'r', encoding='utf-8') as f1, \
        open(prefix_2 + s, 'r', encoding='utf-8') as f2, \
        open(prefix_3 + s, 'r', encoding='utf-8') as f3, \
        open(prefix_save + s, 'w', encoding='utf-8') as fs:
        lines_1 = f1.readlines()
        lines_2 = f2.readlines()
        lines_3 = f3.readlines()
        fs.writelines(lines_1)
        fs.writelines(lines_2)
        fs.writelines(lines_3)
"""

# 两份finetune数据合并到一起啦
prefix_1 = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/dev/dev."
prefix_3 = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_neg_out_of_order/dev/dev."
prefix_save = "/home/yanym/data/qe_merge_real_neg_3o/dev/dev."

def auto_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

auto_mkdir('/'.join(prefix_save.split('/')[0:-1]))

split = ['src', 'src.BPE', 'mt', 'src_tag', 'mt_tag', 'mtgap_tag', 'hter']
for s in split:
    with open(prefix_1 + s, 'r', encoding='utf-8') as f1, \
        open(prefix_3 + s, 'r', encoding='utf-8') as f3, \
        open(prefix_save + s, 'w', encoding='utf-8') as fs:
        lines_1 = f1.readlines()
        lines_3 = f3.readlines()
        fs.writelines(lines_1)
        fs.writelines(lines_3)

# python scripts/contrastive_learning/data_merge.py