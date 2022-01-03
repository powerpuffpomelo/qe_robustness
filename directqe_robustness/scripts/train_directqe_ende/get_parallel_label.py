# 获取预训练平行语料的标注，也就是，全ok，hter全0
# 14.52
#"/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/dev.en_100w.BPE"
#"/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/dev.zh_100w"

# 先在小数据上测试

# 14.52
test_prefix = "/home/yanym/data/parallel_label_ende/"

"""
for file1, file2 in zip(["dev.mt", "dev.tag", "dev.hter"], ["little.mt", "little.tag", "little.hter"] ):
    with open(test_prefix+file1, 'r', encoding='utf-8') as f, open(test_prefix+file2, 'w', encoding='utf-8') as f2:
        lines = f.readlines()[:5]
        f2.writelines(lines)
"""

# 14.59
"""
with open(test_prefix+"little.mt", 'r', encoding='utf-8') as fmt,\
    open(test_prefix+"little.tag1", 'w', encoding='utf-8') as ftag,\
    open(test_prefix+"little.hter1", 'w', encoding='utf-8') as fhter:
    lines = fmt.readlines()
    for line in lines:
        line = line.strip().split()
        tag_list = ['OK'] * len(line)
        ftag.write(" ".join(tag_list) + '\n')
        fhter.write("0.000000\n")
"""

# 15.o4 测试无误

parallel_file = "/home/yanym/data/parallel_label_ende/corpus_train.de.BPE"  # 注意要用tgt端的语料生成label
"""
with open(parallel_file, 'r', encoding='utf-8') as fmt,\
    open(test_prefix+"corpus_train.tag.BPE", 'w', encoding='utf-8') as ftag,\
    open(test_prefix+"corpus_train.hter", 'w', encoding='utf-8') as fhter:
    lines = fmt.readlines()
    for line in lines:
        line = line.strip().split()
        tag_list = ['OK'] * len(line)
        ftag.write(" ".join(tag_list) + '\n')
        fhter.write("0\n")

assert 1==2
"""
# 15.33 合并平行语料和伪数据
def merge_two_file(file1, file2, file3):
    with open(file1, 'r', encoding='utf-8') as f1,\
        open(file2, 'r', encoding='utf-8') as f2,\
        open(file3, 'w', encoding='utf-8') as f3:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        f3.writelines(lines1)
        f3.writelines(lines2)

# dev
merge_two_file(test_prefix+'corpus_dev.en.BPE', test_prefix+'pseudo_dev.src.BPE', test_prefix+'parallel_pseudo_dev.src.BPE')
merge_two_file(test_prefix+'corpus_dev.de.BPE', test_prefix+'pseudo_dev.mt.BPE', test_prefix+'parallel_pseudo_dev.mt.BPE')
merge_two_file(test_prefix+'corpus_dev.tag.BPE', test_prefix+'pseudo_dev.tag.BPE', test_prefix+'parallel_pseudo_dev.tag.BPE')
merge_two_file(test_prefix+'corpus_dev.hter', test_prefix+'pseudo_dev.hter', test_prefix+'parallel_pseudo_dev.hter')

# train
merge_two_file(test_prefix+'corpus_train.en.BPE', test_prefix+'pseudo_train.src.BPE', test_prefix+'parallel_pseudo_train.src.BPE')
merge_two_file(test_prefix+'corpus_train.de.BPE', test_prefix+'pseudo_train.mt.BPE', test_prefix+'parallel_pseudo_train.mt.BPE')
merge_two_file(test_prefix+'corpus_train.tag.BPE', test_prefix+'pseudo_train.tag.BPE', test_prefix+'parallel_pseudo_train.tag.BPE')
merge_two_file(test_prefix+'corpus_train.hter', test_prefix+'pseudo_train.hter', test_prefix+'parallel_pseudo_train.hter')

# python scripts/train_directqe_ende/get_parallel_label.py