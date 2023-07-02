file1 = '/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/train.enzh.zh'
file2 = '/home/data_ti4_c/yanym/data/wmt20/enzh/task2/train/train.mt'

file_save = '/home/yanym/save/data/train.enzh.zh.plus_qe_mt'

with open(file1, 'r', encoding='utf-8') as f1, \
    open(file2, 'r', encoding='utf-8') as f2, \
    open(file_save, 'w', encoding='utf-8') as f3:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    f3.writelines(lines1)
    f3.writelines(lines2)

# python scripts/merge_file.py