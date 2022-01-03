
path_prefix = '/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data/'

for split in ['train', 'dev', 'test']:
    with open(path_prefix+split+'.pe', 'r', encoding='utf-8') as f, \
        open(path_prefix+split+'.lenpe', 'w', encoding='utf-8') as flen:
        for line in f.readlines():
            len_line = len(line.strip('\n').split(' '))
            flen.write(str(len_line)+'\n')

# python scripts/get_len.py