import random
seed = 12138
random.seed(seed)

#test_mt_path = "/home/yanym/save/enzh/src_infl/order/test.mt"
#random_tag_path = "/home/yanym/save/enzh/src_infl/order/test_result_random/test.tag.random_" + str(seed)
test_mt_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/train/train.mt"
random_tag_path = "/home/yanym/save/enzh/src_infl/order/train_result_random/train.tag.random_" + str(seed)

with open(test_mt_path, 'r', encoding='utf-8') as f1, open(random_tag_path, 'w', encoding='utf-8') as f2:
    mt_lines = f1.readlines()
    for line in mt_lines:
        line = line.strip('\n').split()
        tag_line_num = [random.randint(0,1) for i in range(len(line))]
        # 1ok 0bad
        tag_line = ['OK' if tt == 1 else 'BAD' for tt in tag_line_num]
        f2.write(' '.join(tag_line) + '\n')

# python scripts/analysis/src_infl/random_tag.py