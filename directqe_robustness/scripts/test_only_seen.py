# 从test集中，把所有mt词都在训练时见过的样本挑出来

import string
import re

# enzh
data_prefix = "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/"
train_mt_file = "train.mt"
test_mt_file = "test.mt"

# 得到训练词汇集合
train_wordset = set()
with open(data_prefix+train_mt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.lower()
        tokens = re.split(r"[ ]+", line.strip('\n'))
        if tokens == []: continue
        for w in tokens:
            #w = w.strip(string.punctuation)
            if w == '': continue
            train_wordset.add(w)

#print(train_wordset)


# 逐条分析测试样本中的词是否都在训练时出现过
all_in_num = 0
with open(data_prefix+test_mt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.lower()
        tokens = re.split(r"[ ]+", line.strip('\n'))
        all_in_flag = True
        for token in tokens:
            if token not in train_wordset:
                all_in_flag = False
                break
        if all_in_flag == True:
            all_in_num += 1

print(all_in_num)

# python scripts/test_only_seen.py