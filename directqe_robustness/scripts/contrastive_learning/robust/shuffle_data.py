import random

random.seed(1234)

prefix = "/data1/yanym/data/pe_tag/pe_aug_origin_qe_train."

fsrc = open(prefix + 'src.BPE', 'r', encoding='utf-8')
fmt = open(prefix + 'mt.BPE', 'r', encoding='utf-8')
ftag = open(prefix + 'mt_tag.BPE', 'r', encoding='utf-8')
fhter = open(prefix + 'hter', 'r', encoding='utf-8')

fsrc_shuffle = open(prefix + 'src_shuffle.BPE', 'w', encoding='utf-8')
fmt_shuffle = open(prefix + 'mt_shuffle.BPE', 'w', encoding='utf-8')
ftag_shuffle = open(prefix + 'mt_tag_shuffle.BPE', 'w', encoding='utf-8')
fhter_shuffle = open(prefix + 'hter_shuffle', 'w', encoding='utf-8')

src_lines = fsrc.readlines()
mt_lines = fmt.readlines()
tag_lines = ftag.readlines()
hter_lines = fhter.readlines()

id_list = [i for i in range(len(src_lines))]
random.shuffle(id_list)

for id in id_list:
    fsrc_shuffle.write(src_lines[id])
    fmt_shuffle.write(mt_lines[id])
    ftag_shuffle.write(tag_lines[id])
    fhter_shuffle.write(hter_lines[id])


# python scripts/contrastive_learning/robust/shuffle_data.py