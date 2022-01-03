path_prefix = '/data1/yanym/data/robust_qe_train/generate_dev/robust_'
save_prefix = "/data1/yanym/data/toy_train_5/toy5_naive_aug_"

for file in ['train.src.BPE', 'train.mt.BPE', 'train.tag.BPE', 'train.idx.BPE', 'train.hter']:
    with open(path_prefix + file, 'r', encoding='utf-8') as f,\
        open(save_prefix + file, 'w', encoding='utf-8') as toyf:
        lines = f.readlines()
        toyf.writelines(lines[:21])

# python scripts/toy_data.py