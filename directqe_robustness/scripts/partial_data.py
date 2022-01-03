import numpy as np
path_prefix = "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/"
write_path_prefix = "/home/yanym/data_test/"

sample_num = 1000000

with open(path_prefix + "train.enzh.en", 'r', encoding='utf-8') as f1,\
    open(path_prefix + "train.enzh.zh", 'r', encoding='utf-8') as f2,\
    open(write_path_prefix + "train.enzh.en_100w", 'w', encoding='utf-8') as f3,\
    open(write_path_prefix + "train.enzh.zh_100w", 'w', encoding='utf-8') as f4:
    en_all = f1.readlines()
    zh_all = f2.readlines()
    len_all = len(en_all)
    print("len_all" + str(len_all))
    zip_enzh = list(zip(en_all, zh_all))
    sample_ids = np.random.choice(range(len_all), size=sample_num, replace=False)
    for sample_id in sample_ids:
        sample_en, sample_zh = zip_enzh[sample_id]
        f3.writelines(sample_en)
        f4.writelines(sample_zh)

# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/partial_data.py