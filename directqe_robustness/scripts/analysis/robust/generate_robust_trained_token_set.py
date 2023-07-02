# 维护被训练过鲁棒性的token集合，存起来
#"""
mt_bpe_path = "/data1/yanym/data/robust_qe_train_word2vec_double/robust_train.mt.BPE"
robust_idx_path = "/data1/yanym/data/robust_qe_train_word2vec_double/robust_train.idx.BPE"   # 指示哪些mt token是被训练鲁棒性的（即src改变、标签翻转）
robust_token_save_path = "/data1/yanym/data/robust_qe_train_word2vec_double/robust_train.robust_token_set"

with open(mt_bpe_path, 'r', encoding='utf-8') as fmt, open(robust_idx_path, 'r', encoding='utf-8') as fidx, \
    open(robust_token_save_path, 'w', encoding='utf-8') as fset:
    robust_token_save_set = set()
    mt_lines = fmt.readlines()
    idx_lines = fidx.readlines()
    for mt_line, idx_line in zip(mt_lines, idx_lines):
        mt_line = mt_line.strip('\n').split()
        idx_line = list(map(int, (idx_line.strip('\n').split())))
        for idx in idx_line:
            if mt_line[idx] not in robust_token_save_set:
                robust_token_save_set.add(mt_line[idx])
    robust_token_save_set = list(robust_token_save_set)
    for token in robust_token_save_set:
        fset.write(token + '\n')
#"""

# 维护一些训练数据中包含的token集合，存起来
"""
mt_bpe_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt.BPE"
robust_token_save_path = "/data1/yanym/data/normal_train/normal_train.token_set"

with open(mt_bpe_path, 'r', encoding='utf-8') as fmt, \
    open(robust_token_save_path, 'w', encoding='utf-8') as fset:
    robust_token_save_set = set()
    mt_lines = fmt.readlines()[:1200]
    for mt_line in mt_lines:
        mt_line = mt_line.strip('\n').split()
        for token in mt_line:
            if token not in robust_token_save_set:
                robust_token_save_set.add(token)
    robust_token_save_set = list(robust_token_save_set)
    for token in robust_token_save_set:
        fset.write(token + '\n')
"""

# python scripts/analysis/robust/generate_robust_trained_token_set.py