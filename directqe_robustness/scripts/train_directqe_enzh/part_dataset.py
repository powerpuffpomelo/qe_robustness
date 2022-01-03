import numpy as np
import os

size_all = 7000
num_part = 500

part_train_ids = np.random.choice(size_all, num_part, replace=False)
data_src = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/qe_train_pseudo/qe_train_worse_pseudo.src.BPE"
data_mt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/qe_train_pseudo/qe_train_worse_pseudo.mt"
data_tag = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/qe_train_pseudo/qe_train_worse_pseudo.tag"
data_hter = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/qe_train_pseudo/qe_train_worse_pseudo.hter"

part_prefix = '/home/yanym/save/enzh/generate_dev/'
if not os.path.exists(part_prefix):
    os.makedirs(part_prefix, exist_ok=True)
part_src = part_prefix + 'qe_toy_train_worse_pseudo.src.BPE'
part_mt = part_prefix + 'qe_toy_train_worse_pseudo.mt'
part_tag = part_prefix + 'qe_toy_train_worse_pseudo.tag'
part_hter = part_prefix + 'qe_toy_train_worse_pseudo.hter'

def write_part(data_all, data_part, part_train_ids):
    with open(data_all, 'r', encoding='utf-8') as f_all, open(data_part, 'w', encoding='utf-8') as f_part:
        lines = f_all.readlines()
        part_lines = [lines[i] for i in part_train_ids]
        f_part.writelines(part_lines)

for data_all, data_part in zip([data_src, data_mt, data_tag, data_hter], [part_src, part_mt, part_tag, part_hter]):
    write_part(data_all, data_part, part_train_ids)

# python scripts/train_directqe_enzh/part_dataset.py