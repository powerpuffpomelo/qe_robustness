path_prefix = '/home/yanym/data_test/'


with open(path_prefix + "train1.enzh.zh_100w", 'r', encoding='utf-8') as f_all, \
    open(path_prefix+'train.enzh.zh_100w', 'w', encoding='utf-8') as f_train, \
    open(path_prefix+'dev.enzh.zh_100w', 'w', encoding='utf-8') as f_dev, \
    open(path_prefix+'test.enzh.zh_100w', 'w', encoding='utf-8') as f_test:
    lines = f_all.readlines()
    f_dev.writelines(lines[:2000])
    f_test.writelines(lines[2000:3000])
    f_train.writelines(lines[3000:])

# python scripts/split_parallel.py