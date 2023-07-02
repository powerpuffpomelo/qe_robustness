# 使用pe句做数据增强：与其加入平行语料，不如直接使用pe！这里给pe生成tag和hter，全标ok

pe_file = "/data1/yanym/data/pe_tag/train.pe"
pe_tag_file = "/data1/yanym/data/pe_tag/train.pe_tag"
pe_hter_file = "/data1/yanym/data/pe_tag/train.pe_hter"

with open(pe_file, 'r', encoding='utf-8') as fpe, open(pe_tag_file, 'w', encoding='utf-8') as ftag, \
    open(pe_hter_file, 'w', encoding='utf-8') as fhter:
    for pe_line in fpe.readlines():
        pe_line = pe_line.strip('\n').split()
        ftag.write(' '.join(['OK'] * len(pe_line)) + '\n')
        fhter.write('0.0\n')

# python scripts/contrastive_learning/robust/pe_data_aug.py