# 生成robust idx，这里对所有token生成

file_mt = "/data1/yanym/data/robust_train_memory/switch_src_mt/switch_train.mt"
file_idx_all = "/data1/yanym/data/robust_train_memory/switch_src_mt/switch_train.idx"

with open(file_mt, 'r', encoding='utf-8') as fmt, open(file_idx_all, 'w', encoding='utf-8') as fidx:
    mt_lines = fmt.readlines()
    for mt_line in mt_lines:
        fidx.write(" ".join(map(str, list(range(len(mt_line.strip('\n').split()))))) + "\n")

# python scripts/contrastive_learning/robust/get_robust_idx_temp.py