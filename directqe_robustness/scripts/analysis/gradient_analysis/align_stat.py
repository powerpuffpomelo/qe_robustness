# 统计align情况，根据梯度得到的模型预测词对齐结果，和官方词对齐文件比较，
# 统计对一个mt token而言，【影响力最大的src token】有多大比例属于【和它对齐的src token】

import json

"""
align_gold_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_test/test.src-mt.alignments"
align_pred_file = "/home/yanym/save/enzh/gradient/src_align.txt"
# origin2bpe_idx_file = "/home/yanym/save/enzh/gradient/draft/test.src.origin2bpe.idx"
align_gold_bpe_file = "/home/yanym/save/enzh/gradient/draft/test.src-mt.alignments.bpe"

num = 0
ans = 0

with open(align_pred_file, 'r', encoding='utf-8') as fp, \
    open(align_gold_file, 'r', encoding='utf-8') as fg:
    for pred_line, gold_line in zip(fp.readlines(), fg.readlines()):
        pred_list = pred_line.strip('\n').split()
        pred_list = [int(item) for item in pred_list]
        gold_list = gold_line.strip('\n').split()
        #print(pred_list)
        #print(gold_list)
        gold_dict_line = dict()
        for gold_item in gold_list:
            src_idx, mt_idx = gold_item.split('-')
            src_idx = int(src_idx)
            mt_idx = int(mt_idx)
            if mt_idx not in gold_dict_line: gold_dict_line[mt_idx] = []
            gold_dict_line[mt_idx].append(src_idx)
        for mt_idx in range(len(pred_list)):
            if mt_idx not in gold_dict_line: continue    # 没有对齐的排除
            if pred_list[mt_idx] == -1: continue    # unk排除
            most_att = pred_list[mt_idx]     # 预测最对齐的src
            if most_att in gold_dict_line[mt_idx]: ans += 1
            num += 1

print("align_token_num = %d" % ans)
print("all_consider_num = %d" % num)
print("align_ratio = %.6f" % (ans / num))
"""

# =================================================================================================================== #
# align_file2bpe
# 先根据官方对齐文件，找到一个mt token对应的所有src token；再根据ori2bpe文件，把src token索引转化成bpe格式，存起来

# wmt20 enzh
align_gold_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/train/train.src-mt.alignments"
src_origin_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/train/train.src"
src_bpe_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/train/train.src.BPE"
align_gold_bpe_file = "/home/yanym/data/train.src-mt.alignments.BPE"
mt_origin_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/train/train.mt"
mt_bpe_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/train/train.mt.BPE"

"""
# wmt20 enzh train
align_gold_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.src-mt.alignments"
src_origin_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.src"
src_bpe_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.src.BPE"
align_gold_bpe_file = "/home/yanym/save/directqe_enzh/gradient/draft/train.src-mt.alignments.bpe"

# wmt21 enzh
align_gold_file = "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.src-mt.alignments"
src_origin_file = "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.src"
src_bpe_file = "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_test/test.src.BPE"
align_gold_bpe_file = "/home/yanym/save/enzh/gradient/draft/test.src-mt.alignments.bpe21"

# wmt20 ende
align_gold_file = "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version_transquest/qe_test/test.src-mt.alignments"
src_origin_file = "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version_transquest/qe_test/test.src"
src_bpe_file = "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version/test.src.BPE"
align_gold_bpe_file = "/data1/yanym/save/ende/gradient/draft/test.src-mt.alignments.bpe"
mt_origin_file = "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version_transquest/qe_test/test.mt"
mt_bpe_file = "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version/test.mt.BPE"
"""

def align_file2bpe(align_gold_file, src_origin_file, src_bpe_file, align_gold_bpe_file, mt_origin_file=None, mt_bpe_file=None):
    # 先根据官方对齐文件，找到一个mt token对应的所有src token；再根据ori2bpe文件，把src token索引转化成bpe格式，存起来
    if mt_origin_file is None:
        with open(align_gold_file, 'r', encoding='utf-8') as fag, \
            open(src_origin_file, 'r', encoding='utf-8') as fori, \
            open(src_bpe_file, 'r', encoding='utf-8') as fbpe, \
            open(align_gold_bpe_file, 'w', encoding='utf-8') as fagbpe:

            idx_bpe_dict = dict()
            line_id = 0

            for ag_line, ori_line, bpe_line in zip(fag.readlines(), fori.readlines(), fbpe.readlines()):
                ag_list = ag_line.strip('\n').split()
                ori_line = ori_line.strip('\n').split()
                bpe_line = bpe_line.strip('\n').split()

                # src每个完整词汇对应的bpe位置
                idx_line = dict()
                for i in range(len(ori_line)):
                    idx_line[i] = []
                i = 0
                for j, tt in enumerate(bpe_line):
                    idx_line[i].append(j)
                    if "@@" not in tt:
                        i += 1
                
                # mt每个词汇对齐的src词汇对应的bpe位置
                idx_bpe_line = dict()
                for ag_item in ag_list:
                    src_idx, mt_idx = ag_item.split('-')
                    src_idx = int(src_idx)
                    src_idx_bpe = idx_line[src_idx]
                    mt_idx = int(mt_idx)
                    if mt_idx not in idx_bpe_line: idx_bpe_line[mt_idx] = []
                    idx_bpe_line[mt_idx].extend(src_idx_bpe)
                idx_bpe_dict[line_id] = idx_bpe_line
                
                line_id += 1

            json.dump(idx_bpe_dict, fagbpe, indent=1)
    else:
        with open(align_gold_file, 'r', encoding='utf-8') as fag, \
            open(src_origin_file, 'r', encoding='utf-8') as fsrc_ori, \
            open(src_bpe_file, 'r', encoding='utf-8') as fsrc_bpe, \
            open(mt_origin_file, 'r', encoding='utf-8') as fmt_ori, \
            open(mt_bpe_file, 'r', encoding='utf-8') as fmt_bpe, \
            open(align_gold_bpe_file, 'w', encoding='utf-8') as fagbpe:

            idx_bpe_dict = dict()
            line_id = 0

            for ag_line, src_ori_line, src_bpe_line, mt_ori_line, mt_bpe_line in zip(fag.readlines(), fsrc_ori.readlines(), fsrc_bpe.readlines(), fmt_ori.readlines(), fmt_bpe.readlines()):
                ag_list = ag_line.strip('\n').split()
                src_ori_line = src_ori_line.strip('\n').split()
                src_bpe_line = src_bpe_line.strip('\n').split()
                mt_ori_line = mt_ori_line.strip('\n').split()
                mt_bpe_line = mt_bpe_line.strip('\n').split()

                # src每个完整词汇对应的bpe位置
                src_ori2bpe_line = dict()
                for i in range(len(src_ori_line)):
                    src_ori2bpe_line[i] = []
                i = 0
                for j, tt in enumerate(src_bpe_line):
                    src_ori2bpe_line[i].append(j)
                    if "@@" not in tt:
                        i += 1

                # mt每个完整词汇对应的bpe位置
                mt_ori2bpe_line = dict()
                for i in range(len(mt_ori_line)):
                    mt_ori2bpe_line[i] = []
                i = 0
                for j, tt in enumerate(mt_bpe_line):
                    mt_ori2bpe_line[i].append(j)
                    if "@@" not in tt:
                        i += 1

                # mt每个词汇对齐的src词汇对应的bpe位置
                idx_bpe_line = dict()
                for ag_item in ag_list:
                    src_idx, mt_idx = ag_item.split('-')
                    src_idx = int(src_idx)
                    src_idx_bpe = src_ori2bpe_line[src_idx]
                    mt_idx = int(mt_idx)
                    if mt_idx not in idx_bpe_line: idx_bpe_line[mt_idx] = []
                    idx_bpe_line[mt_idx].extend(src_idx_bpe)   # 一个mt word对应的每个src token，但没考虑重复的情况！应该用set
                new_idx_bpe_line = dict()
                for mt_idx in idx_bpe_line:
                    mt_idx_bpe_list = mt_ori2bpe_line[mt_idx]
                    for mt_idx_bpe in mt_idx_bpe_list:
                        new_idx_bpe_line[mt_idx_bpe] = idx_bpe_line[mt_idx]   # 每个mt bpe token，都对齐原来mt word对齐的内容，都复制一份。
                idx_bpe_dict[line_id] = new_idx_bpe_line
                
                line_id += 1

            json.dump(idx_bpe_dict, fagbpe, indent=1)

align_file2bpe(align_gold_file, src_origin_file, src_bpe_file, align_gold_bpe_file, mt_origin_file, mt_bpe_file)

# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/analysis/gradient_analysis/align_stat.py