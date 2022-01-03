"""
align_file2bpe
需要对齐文件，有src word和tgt word的对应关系
对于每个src token，找到其所在的src word；然后找到该src word对齐的全部tgt word，这些tgt word包含的tgt token集合，就是这一个src token维护的集合。
所以本文件是单向的
"""

import json
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--align_gold_file', type=str)
parser.add_argument('--src_origin_file', type=str)
parser.add_argument('--src_bpe_file', type=str)
parser.add_argument('--tgt_origin_file', type=str)
parser.add_argument('--tgt_bpe_file', type=str)
parser.add_argument('--align_gold_bpe_file', type=str)

args = parser.parse_args()

# wmt20 enzh
align_gold_file = args.align_gold_file
src_origin_file = args.src_origin_file
src_bpe_file = args.src_bpe_file
tgt_origin_file = args.tgt_origin_file
tgt_bpe_file = args.tgt_bpe_file
align_gold_bpe_file = args.align_gold_bpe_file


def align_file2bpe(align_gold_file, src_origin_file, src_bpe_file, tgt_origin_file, tgt_bpe_file, align_gold_bpe_file):
    # 先根据官方对齐文件，找到一个tgt token对应的所有src token；再根据ori2bpe文件，把src token索引转化成bpe格式，存起来
    with open(align_gold_file, 'r', encoding='utf-8') as fag, \
        open(src_origin_file, 'r', encoding='utf-8') as fsrc_ori, \
        open(src_bpe_file, 'r', encoding='utf-8') as fsrc_bpe, \
        open(tgt_origin_file, 'r', encoding='utf-8') as ftgt_ori, \
        open(tgt_bpe_file, 'r', encoding='utf-8') as ftgt_bpe, \
        open(align_gold_bpe_file, 'w', encoding='utf-8') as fagbpe:

        idx_bpe_dict = dict()
        line_id = 0

        for ag_line, src_ori_line, src_bpe_line, tgt_ori_line, tgt_bpe_line in zip(fag.readlines(), fsrc_ori.readlines(), fsrc_bpe.readlines(), ftgt_ori.readlines(), ftgt_bpe.readlines()):
            ag_list = ag_line.strip('\n').split()
            src_ori_line = src_ori_line.strip('\n').split()
            src_bpe_line = src_bpe_line.strip('\n').split()
            tgt_ori_line = tgt_ori_line.strip('\n').split()
            tgt_bpe_line = tgt_bpe_line.strip('\n').split()

            # src每个完整词汇对应的bpe位置
            src_ori2bpe_line = dict()
            for i in range(len(src_ori_line)):
                src_ori2bpe_line[i] = []
            i = 0
            for j, tt in enumerate(src_bpe_line):
                src_ori2bpe_line[i].append(j)
                if "@@" not in tt:
                    i += 1

            # tgt每个完整词汇对应的bpe位置
            tgt_ori2bpe_line = dict()
            for i in range(len(tgt_ori_line)):
                tgt_ori2bpe_line[i] = []
            i = 0
            for j, tt in enumerate(tgt_bpe_line):
                tgt_ori2bpe_line[i].append(j)
                if "@@" not in tt:
                    i += 1

            # src每个token所在的词汇对齐的tgt词汇对应的bpe位置集合
            idx_bpe_line = dict()
            for ag_item in ag_list:
                src_idx, tgt_idx = ag_item.split('-')
                src_idx = int(src_idx)
                tgt_idx = int(tgt_idx)
                src_idx_bpe_list = src_ori2bpe_line[src_idx]
                tgt_idx_bpe_list = tgt_ori2bpe_line[tgt_idx]
                for src_idx_bpe in src_idx_bpe_list:
                    if src_idx_bpe not in idx_bpe_line: idx_bpe_line[src_idx_bpe] = set()
                    idx_bpe_line[src_idx_bpe].update(tgt_idx_bpe_list)

            # 集合转列表
            for k in idx_bpe_line:
                idx_bpe_line[k] = list(idx_bpe_line[k])

            idx_bpe_dict[line_id] = idx_bpe_line
            line_id += 1

        json.dump(idx_bpe_dict, fagbpe, indent=1)

align_file2bpe(align_gold_file, src_origin_file, src_bpe_file, tgt_origin_file, tgt_bpe_file, align_gold_bpe_file)

# python scripts/analysis/gradient_analysis/align_file2bpe.py