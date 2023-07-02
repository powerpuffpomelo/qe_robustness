prefix1 = '/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.'
prefix2 = '/data1/yanym/data/pe_tag/train.'

prefix_save = '/data1/yanym/data/pe_tag/pe_aug_origin_qe_train.'

file_list = ['src', 'mt', 'mt_tag', 'hter']

for file in file_list:
    with open(prefix1 + file, 'r', encoding='utf-8') as f1, \
        open(prefix2 + file, 'r', encoding='utf-8') as f2, \
        open(prefix_save + file, 'w', encoding='utf-8') as f3:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        f3.writelines(lines1)
        f3.writelines(lines2)

# python scripts/contrastive_learning/robust/merge_file.py