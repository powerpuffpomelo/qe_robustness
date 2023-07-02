# 之前的robust idx 标明了 原始+增强 中 改变的部分
# 这里删掉增强的部分，只留下原始数据的idx部分，就是说，这是普通qe测试集和鲁棒性测试集的交集
"""
file_mt = "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/robust_test.mt"
file_idx_ori = "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/robust_test.idx.BPE"
file_idx_com = "/data1/yanym/data/robust_qe_test_word2vec_limit_3_10/robust_test.idx_common.BPE"

with open(file_mt, 'r', encoding='utf-8') as fmt, open(file_idx_ori, 'r', encoding='utf-8') as fori, \
    open(file_idx_com, 'w', encoding='utf-8') as fcom:
    mt_lines = fmt.readlines()
    idx_ori_lines = fori.readlines()
    i = 0
    while i < len(mt_lines):
        print(i)
        if i == 0 or mt_lines[i] != mt_lines[i - 1]:  # 开启一个新的样本
            fcom.write(idx_ori_lines[i])
        else:
            fcom.write('\n')
        i += 1
"""

file_mt = "/data1/yanym/data/robust_word_alignment/robust_test.mt"
file_idx_ori = "/data1/yanym/data/robust_word_alignment/robust_test.idx.BPE"
file_idx_com = "/data1/yanym/data/robust_word_alignment/robust_test.idx_common.BPE"

with open(file_mt, 'r', encoding='utf-8') as fmt, open(file_idx_ori, 'r', encoding='utf-8') as fori, \
    open(file_idx_com, 'w', encoding='utf-8') as fcom:
    mt_lines = fmt.readlines()
    idx_ori_lines = fori.readlines()
    i = 0
    while i < len(mt_lines):
        print(i)
        if i == len(mt_lines) - 1 or mt_lines[i] != mt_lines[i + 1]:  # 一个样本的收尾
            fcom.write(idx_ori_lines[i])
        else:
            fcom.write('\n')
        i += 1

# python scripts/contrastive_learning/robust/idx_robust_common.py