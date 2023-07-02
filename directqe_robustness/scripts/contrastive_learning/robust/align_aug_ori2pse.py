# 扩充对齐文档到伪数据
本代码没写完！！！！

import json

mt_file = "/data1/yanym/data/robust_language_model/robust_qe_train_word2vec_double/robust_train.mt"
align_ori = "/data1/yanym/data/test_align_enzh/train.mt-src.alignments.BPE"
align_pse = "/data1/yanym/data/robust_language_model/robust_qe_train_word2vec_double/robust_train.mt-src.alignments.BPE.json"

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

# python scripts/contrastive_learning/robust/align_aug_ori2pse.py