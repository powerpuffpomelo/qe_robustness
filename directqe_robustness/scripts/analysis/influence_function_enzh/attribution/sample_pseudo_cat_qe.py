import numpy as np

pseudo_prefix = "/home/user_data_182b/yanym/qe/data/wmt19_ende_pseudo_from_train/"
qe_prefix = "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/"

# 来个toy的先
# pseudo_files = ['mlm_dev.src.BPE', 'mlm_dev.mt.BPE', 'mlm_dev.tag', 'mlm_dev.hter']
pseudo_files = ['pseudo.src.BPE', 'pseudo.mt.BPE', 'pseudo.tag', 'pseudo.hter']
# qe_files = ['toy_train.src.BPE', 'toy_train.mt.BPE', 'toy_train.tags.BPE', 'toy_train.hter']
qe_files = ['train.src.BPE', 'train.mt.BPE', 'train.tags.BPE', 'train.hter']

# 采样，生成采样文件
time = 1       # 第几次采样
num_w = 1      # 从340w中采样多少w
sample_info = 'sample' + str(time) + '_' + str(num_w) + 'w.'
sample_files = [sample_info + p for p in pseudo_files]
cat_info = 'cat' + str(time) + '_' + str(num_w) + 'w.'
cat_files = [cat_info + p for p in pseudo_files]

with open(pseudo_prefix + pseudo_files[0], 'r', encoding='utf-8') as fsrc, open(pseudo_prefix + pseudo_files[1], 'r', encoding='utf-8') as fmt, \
    open(pseudo_prefix + pseudo_files[2], 'r', encoding='utf-8') as ftag, open(pseudo_prefix + pseudo_files[3], 'r', encoding='utf-8') as fhter, \
    open(pseudo_prefix + sample_files[0], 'w', encoding='utf-8') as ssrc, open(pseudo_prefix + sample_files[1], 'w', encoding='utf-8') as smt, \
    open(pseudo_prefix + sample_files[2], 'w', encoding='utf-8') as stag, open(pseudo_prefix + sample_files[3], 'w', encoding='utf-8') as shter:
    src = fsrc.readlines()
    mt = fmt.readlines()
    tag = ftag.readlines()
    hter = fhter.readlines()
    train_dataset = list(zip(src,mt,tag,hter))
    sample_ids = np.random.choice(range(len(train_dataset)), size=num_w*10000, replace=False)
    for sample_id in sample_ids:
        src_line, mt_line, tag_line, hter_line = train_dataset[sample_id]
        ssrc.write(src_line)
        smt.write(mt_line)
        stag.write(tag_line)
        shter.write(hter_line)

# 拼接，生成 采样的伪数据 和 qe训练数据 结合的文件
for i in range(4):
    with open(pseudo_prefix + sample_files[i], 'r', encoding='utf-8') as f1, open(qe_prefix + qe_files[i], 'r', encoding='utf-8') as f2, \
        open(pseudo_prefix + cat_files[i], 'w', encoding='utf-8') as fcat:
        f1_lines = f1.readlines()
        f2_lines = f2.readlines()
        fcat.writelines(f1_lines)
        fcat.writelines(f2_lines)

# python scripts/attribution/sample_pseudo_cat_qe.py