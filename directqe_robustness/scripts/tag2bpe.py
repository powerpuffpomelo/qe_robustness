#"""
bpe_file = '/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.mt.BPE'
original_tag = '/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.mt_tag'
bpe_tag = '/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.mt_tag.BPE'

with open(bpe_file, 'r', encoding='utf-8') as f1, open(original_tag, 'r', encoding='utf-8') as f2, \
                                                    open(bpe_tag, 'w', encoding='utf-8') as f3:
    line_id = 0
    for f1_line, f2_line in zip(f1,f2):
        #print(line_id)
        bpe_tag_line = []
        bpe_file_line = f1_line.strip('\n').split()[::-1]
        tag_line = f2_line.strip('\n').split()[::-1]
        id = 0
        for bpe_word in bpe_file_line:
            if '@@' not in bpe_word:
                bpe_tag_line.append(tag_line[id])
                id += 1
            else:
                bpe_tag_line.append(tag_line[id-1])
        bpe_tag_line = ' '.join(bpe_tag_line[::-1]) + '\n'
        f3.write(bpe_tag_line)
        line_id += 1
#"""
"""
# ================================== tag_bpe2origin 把bpe版本的tag还原 ================================== #
bpe_file = "/data1/yanym/data/robust_qe_test_ende/generate_dev/robust_test.mt.BPE"
bpe_tag = "/data1/yanym/data/robust_qe_test_ende/generate_dev/robust_test.tag.BPE"
original_tag = "/data1/yanym/data/robust_qe_test_ende/generate_dev/robust_test.tag"

with open(bpe_file, 'r', encoding='utf-8') as f1, open(bpe_tag, 'r', encoding='utf-8') as f2, \
                                                    open(original_tag, 'w', encoding='utf-8') as f3:
    for f1_line, f2_line in zip(f1, f2):
        origin_tag_line = []
        bpe_file_line = f1_line.strip('\n').split()
        bpe_tag_line = f2_line.strip('\n').split()
        id = 0
        for bpe_token in bpe_file_line:
            if id == 0 or '@@' not in bpe_file_line[id - 1]:
                origin_tag_line.append(bpe_tag_line[id])
            id += 1
        origin_tag_line = ' '.join(origin_tag_line) + '\n'
        f3.write(origin_tag_line)

"""

# python scripts/tag2bpe.py