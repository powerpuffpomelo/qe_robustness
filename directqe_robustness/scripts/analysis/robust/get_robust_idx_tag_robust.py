# 只得到idx为1处的预测tag存起来，其余tag不care
# 鲁棒性idx也要考虑，就是说同时考虑鲁棒性测试集要测试的token idx，以及在qe鲁棒性训练集训练过鲁棒性的token idx

all_tag_path = "/data1/yanym/data/robust_qe_test_analysis/qe_robust_test_result/v1_test_all.tag.BPE"
robust_idx_path = "/data1/yanym/data/robust_qe_test_analysis/robust_test_robust_trained_idx.BPE"  # 在qe鲁棒性训练集中训练过鲁棒性的token idx
robust_test_idx_path = "/data1/yanym/data/robust_qe_test/generate_dev/robust_test.idx.BPE"  # 鲁棒性测试集要测试的token idx
robust_tag_path = "/data1/yanym/data/robust_qe_test_analysis/qe_robust_test_result/v1_not_robust.tag_pred.BPE"

with open(all_tag_path, 'r', encoding='utf-8') as ft, open(robust_idx_path, 'r', encoding='utf-8') as fid, \
    open(robust_test_idx_path, 'r', encoding='utf-8') as froid, open(robust_tag_path, 'w', encoding='utf-8') as fsave:
    all_tag_lines = ft.readlines()
    idx_lines = fid.readlines()
    ro_idx_lines = froid.readlines()
    for all_tag_line, idx_line, ro_idx_line in zip(all_tag_lines, idx_lines, ro_idx_lines):
        all_tag_line = all_tag_line.strip('\n').split()
        idx_line = list(map(int, idx_line.strip('\n').split()))
        ro_idx_line = list(map(int, ro_idx_line.strip('\n').split()))
        robust_tag_line = [all_tag_line[i] for i in range(len(idx_line)) if i in ro_idx_line and idx_line[i] == 0]
        fsave.write(' '.join(robust_tag_line) + '\n')

# python scripts/analysis/robust/get_robust_idx_tag_robust.py