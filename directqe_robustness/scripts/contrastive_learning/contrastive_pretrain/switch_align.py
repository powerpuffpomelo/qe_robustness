# 反转align文件

align_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_real/test/test.src-mt.alignments"
switch_align_file = "/home/yanym/test.mt-src.alignments"

with open(align_file, 'r', encoding='utf-8') as f1, open(switch_align_file, 'w', encoding='utf-8') as f2:
    for line in f1.readlines():
        line = line.strip('\n').split()
        print(line)
        new_line = []
        for item in line:
            src_id, mt_id = item.split('-')
            new_line.append(mt_id + '-' + src_id) 
        f2.write(' '.join(new_line) + '\n')

# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/contrastive_learning/contrastive_pretrain/switch_align.py