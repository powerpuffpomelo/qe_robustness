file1 = '/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/train.enzh.zh'
file2 = '/home/data_ti4_c/yanym/data/wmt20/enzh/task2/train/train.mt'

#file1 = '/home/yanym/save/data/aa'
#file2 = '/home/yanym/save/data/bb'
file_save = '/home/yanym/save/data/train.enzh.zh.plus_qe_mt'

with open(file1, 'r', encoding='utf-8') as f1, \
    open(file2, 'r', encoding='utf-8') as f2, \
    open(file_save, 'w', encoding='utf-8') as f3:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    for i in range(100):
        f3.writelines(lines2)
    f3.writelines(lines1)

# python scripts/merge_file.py