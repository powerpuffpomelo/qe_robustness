# 得到错位的训练集src mt 和全bad的tag 文件

qe_train_src = "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_train/train.src"
qe_train_mt = "/home/data_ti4_c/yanym/data/wmt21/post-editing/mello_version_enzh/qe_train/train.mt"

switch_train_src = "/data1/yanym/data/robust_train_memory_wmt21/switch_src_mt/switch_train.src"
switch_train_mt = "/data1/yanym/data/robust_train_memory_wmt21/switch_src_mt/switch_train.mt"
switch_train_mt_tag = "/data1/yanym/data/robust_train_memory_wmt21/switch_src_mt/switch_train.mt_tag"
switch_train_hter = "/data1/yanym/data/robust_train_memory_wmt21/switch_src_mt/switch_train.hter"
switch_train_mt_gap_tag = "/data1/yanym/data/robust_train_memory_wmt21/switch_src_mt/switch_train.mt_gap_tag"
switch_train_src_tag = "/data1/yanym/data/robust_train_memory_wmt21/switch_src_mt/switch_train.src_tag"

with open(qe_train_src, 'r', encoding='utf-8') as f_ori_src, open(qe_train_mt, 'r', encoding='utf-8') as f_ori_mt, \
    open(switch_train_src, 'w', encoding='utf-8') as f_swi_src, open(switch_train_mt, 'w', encoding='utf-8') as f_swi_mt, \
    open(switch_train_mt_tag, 'w', encoding='utf-8') as f_swi_mt_tag, open(switch_train_hter, 'w', encoding='utf-8') as f_swi_hter, \
    open(switch_train_mt_gap_tag, 'w', encoding='utf-8') as f_swi_mt_gap_tag, open(switch_train_src_tag, 'w', encoding='utf-8') as f_swi_src_tag:
    move_k = 500
    ori_src_lines = f_ori_src.readlines()
    ori_src_lines = ori_src_lines[move_k:] + ori_src_lines[:move_k]
    f_swi_src.writelines(ori_src_lines)
    ori_mt_lines = f_ori_mt.readlines()
    f_swi_mt.writelines(ori_mt_lines)
    for src_line, mt_line in zip(ori_src_lines, ori_mt_lines):
        src_line = src_line.strip('\n').split()
        mt_line = mt_line.strip('\n').split()
        swi_src_tag_line = ["BAD" for i in range(len(src_line))]
        swi_mt_tag_line = ["BAD" for i in range(len(mt_line))]
        swi_mt_gap_tag_line = swi_mt_tag_line + swi_mt_tag_line + ['BAD']
        f_swi_src_tag.write(' '.join(swi_src_tag_line) + '\n')
        f_swi_mt_tag.write(' '.join(swi_mt_tag_line) + '\n')
        f_swi_mt_gap_tag.write(' '.join(swi_mt_gap_tag_line) + '\n')
        f_swi_hter.write('0\n')

# python scripts/contrastive_learning/robust/switch_src_mt.py