# 翻转一个align文件，从src-mt变成mt-src
import json

origin_align_file = "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.src-mt.alignments"
switch_align_file = "/data1/yanym/data/test.mt-src.alignments"

with open(origin_align_file, 'r', encoding='utf-8') as f1, open(switch_align_file, 'w', encoding='utf-8') as f2:
    ori_lines = f1.readlines()
    for ori_line in ori_lines:
        ori_line = ori_line.strip('\n').split()
        swi_line = []
        for item in ori_line:
            src_id, mt_id = item.split('-')
            swi_line.append(mt_id + '-' + src_id)
        f2.write(' '.join(swi_line) + '\n')

# python scripts/analysis/gradient_analysis/align_switch.py