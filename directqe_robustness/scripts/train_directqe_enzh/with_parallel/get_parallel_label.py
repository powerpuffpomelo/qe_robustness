# 获取预训练平行语料的标注，也就是，全ok，hter全0
# 14.52
#"/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/train.en_100w.BPE"
#"/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/train.zh_100w"

# 先在小数据上测试

# 14.52
test_prefix = "/home/yanym/data/"

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

parallel_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/train.zh_100w"

with open(parallel_file, 'r', encoding='utf-8') as fmt,\
    open(test_prefix+"parallel.tag", 'w', encoding='utf-8') as ftag,\
    open(test_prefix+"parallel.hter", 'w', encoding='utf-8') as fhter:
    lines = fmt.readlines()
    for line in lines:
        line = line.strip().split()
        tag_list = ['OK'] * len(line)
        ftag.write(" ".join(tag_list) + '\n')
        fhter.write("0\n")

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

merge_two_file(test_prefix+'train.en_100w.BPE', test_prefix+'pseudo_100w.src.BPE', test_prefix+'parallel_pseudo_100w.src.BPE')
merge_two_file(test_prefix+'train.zh_100w', test_prefix+'pseudo_100w.mt', test_prefix+'parallel_pseudo_100w.mt')
merge_two_file(test_prefix+'parallel.tag', test_prefix+'pseudo_100w.tag', test_prefix+'parallel_pseudo_100w.tag')
merge_two_file(test_prefix+'parallel.hter', test_prefix+'pseudo_100w.hter', test_prefix+'parallel_pseudo_100w.hter')
"""
# python scripts/train_directqe_enzh/with_parallel/get_parallel_label.py