# bpe和origin文件token索引互转

import json
origin_file = "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.src"
bpe_file = "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.src.BPE"
bpe2origin_idx_file = "/home/yanym/save/enzh/gradient_wmt21/test.src.bpe2origin.idx"
origin2bpe_idx_file = "/home/yanym/save/enzh/gradient/draft/test.src.origin2bpe.idx"

# 根据bpe文件生成一个新的索引文件，标明每个bpe token对应到原始文件中的位置

with open(bpe_file, 'r', encoding='utf-8') as fbpe, open(bpe2origin_idx_file, 'w', encoding='utf-8') as fidx:
    bpe_lines = fbpe.readlines()
    for bpe_line in bpe_lines:
        bpe_line = bpe_line.strip('\n').split()
        ss = "0"
        i = 0
        for bpe_token in bpe_line[:-1]:
            if '@@' not in bpe_token:
                i += 1
            ss += (' ' + str(i))
        ss += '\n'
        fidx.write(ss)


# 根据bpe文件生成一个新的索引文件，标明每个原始文件token对应到bpe文件中的位置
"""
with open(origin_file, 'r', encoding='utf-8') as fori, open(bpe_file, 'r', encoding='utf-8') as fbpe, \
    open(origin2bpe_idx_file, 'w', encoding='utf-8') as fidx:
    idx_dict = dict()
    ori_lines = fori.readlines()
    bpe_lines = fbpe.readlines()
    line_id = 0
    for ori_line, bpe_line in zip(ori_lines, bpe_lines):
        ori_line = ori_line.strip('\n').split()
        bpe_line = bpe_line.strip('\n').split()
        idx_line = dict()
        for i in range(len(ori_line)):
            idx_line[i] = []
        i = 0
        for j, tt in enumerate(bpe_line):
            idx_line[i].append(j)
            if "@@" not in tt:
                i += 1
        
        idx_dict[line_id] = idx_line
        line_id += 1

    json.dump(idx_dict, fidx, indent=1)
"""

# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/analysis/gradient_analysis/bpe_indices.py