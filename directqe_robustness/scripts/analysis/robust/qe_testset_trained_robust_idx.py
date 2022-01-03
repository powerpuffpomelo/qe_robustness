# 被训练过鲁棒性的token集合文件，标注qe测试集每个token是否被训练过鲁棒性，标注01，得到一个和qe测试集mt bpe文件对应的idx文件；

qe_test_mt_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE"
robust_token_path = "/data1/yanym/data/robust_qe_train_v1/robust_train.robust_token_set"
qe_test_robust_trained_idx_path = "/data1/yanym/data/robust_qe_test_analysis/qe_test_robust_trained_idx.BPE"

num_one = 0
num_zero = 0

with open(qe_test_mt_path, 'r', encoding='utf-8') as fmt, open(robust_token_path, 'r', encoding='utf-8') as frobust, \
    open(qe_test_robust_trained_idx_path, 'w', encoding='utf-8') as fidx:
    robust_trained_token_set = set()
    for robust_token_line in frobust.readlines():
        robust_trained_token_set.add(robust_token_line.strip('\n'))
    for mt_line in fmt.readlines():
        mt_line = mt_line.strip('\n').split()
        robust_sign_line = [1 if mt_line[i] in robust_trained_token_set else 0 for i in range(len(mt_line))]
        num_one += robust_sign_line.count(1)
        num_zero += robust_sign_line.count(0)
        robust_sign_line = list(map(str, robust_sign_line))
        fidx.write(' '.join(robust_sign_line) + '\n')

print('num of robust_trained tokens: %d' % num_one)
print('num of robust_untrained tokens: %d' % num_zero)
print('ratio of robust trained: %.6f' % (num_one / (num_one + num_zero)))

# python scripts/analysis/robust/qe_testset_trained_robust_idx.py