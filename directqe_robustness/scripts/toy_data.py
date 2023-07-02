path_prefix = '/data1/yanym/data/robust_qe_train_v1/robust_train.'
save_prefix = "/data1/yanym/data/robust_qe_train_v1/toy/toy_robust_train."

for file in ['src.BPE', 'mt.BPE', 'tag.BPE', 'idx.BPE', 'hter']:
    with open(path_prefix + file, 'r', encoding='utf-8') as f,\
        open(save_prefix + file, 'w', encoding='utf-8') as toyf:
        lines = f.readlines()
        toyf.writelines(lines[:5])

# python scripts/toy_data.py