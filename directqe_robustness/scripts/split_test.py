path_prefix = '/home/yanym/data_temp/task2/mello_version/part_train/'
for file in ['hter','mt','src.BPE','tag']:
    with open(path_prefix+'train.'+file, 'r', encoding='utf-8') as f1, \
        open(path_prefix+'train1.'+file, 'w', encoding='utf-8') as f2, \
        open(path_prefix+'train2.'+file, 'w', encoding='utf-8') as f3:
        lines = f1.readlines()
        f2.writelines(lines[:3500])
        f3.writelines(lines[3500:])

# python scripts/split_test.py