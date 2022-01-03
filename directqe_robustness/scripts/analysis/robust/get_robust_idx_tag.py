# 只得到idx为1处的预测tag存起来，其余tag不care

all_tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag.BPE"
robust_idx_path = "/data1/yanym/data/robust_qe_test_analysis/qe_test_robust_trained_idx.BPE"
robust_tag_path = "/data1/yanym/data/robust_qe_test_analysis/result/robust.tag_gold.BPE"

with open(all_tag_path, 'r', encoding='utf-8') as ft, open(robust_idx_path, 'r', encoding='utf-8') as fid, \
    open(robust_tag_path, 'w', encoding='utf-8') as fsave:
    all_tag_lines = ft.readlines()
    idx_lines = fid.readlines()
    for all_tag_line, idx_line in zip(all_tag_lines, idx_lines):
        all_tag_line = all_tag_line.strip('\n').split()
        idx_line = list(map(int, idx_line.strip('\n').split()))
        robust_tag_line = [all_tag_line[i] for i in range(len(idx_line)) if idx_line[i] == 1]
        fsave.write(' '.join(robust_tag_line) + '\n')

# python scripts/analysis/robust/get_robust_idx_tag.py