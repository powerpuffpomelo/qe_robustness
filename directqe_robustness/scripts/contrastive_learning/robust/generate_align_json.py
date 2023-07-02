import json
data_prefix = "/data1/yanym/data/robust_train_memory/align/"
align_mt2src = data_prefix + "train.mt-src.alignments"
align_json = align_mt2src + ".json"

ret_dict = dict()
line_id = 0
with open(align_mt2src, 'r', encoding='utf-8') as falign, \
    open(align_json, 'w', encoding='utf-8') as fj:
    align_mt2src_lines = falign.readlines()
    for align_line in align_mt2src_lines:
        align_list = align_line.strip('\n').split()
        line_dict = dict()
        for align_item in align_list:
            mt_idx, src_idx = align_item.split('-')
            mt_idx = int(mt_idx)
            src_idx = int(src_idx)
            if mt_idx not in line_dict: line_dict[mt_idx] = []
            line_dict[mt_idx].append(src_idx)
        ret_dict[line_id] = line_dict
        line_id += 1
    json.dump(ret_dict, fj, indent = 1)

# python scripts/contrastive_learning/robust/generate_align_json.py