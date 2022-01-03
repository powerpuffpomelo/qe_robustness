import random
seed = 12138
random.seed(seed)

file_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_test/test.mt"
file_3o = "/home/yanym/save/transquest/mt_infl_order/test.mt.3o_token/test.mt.3o_" + str(seed)

with open(file_path, 'r', encoding='utf-8') as f1, open(file_3o, 'w', encoding='utf-8') as f2:
    lines = f1.readlines()
    for line in lines:
        line = line.strip('\n').split()
        random.shuffle(line)
        f2.write(' '.join(line) + '\n')

# bpe
"""
subword-nmt apply-bpe -c /home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/bpe.30000.en < /home/yanym/save/enzh/src_infl/order/train.src.3o_token/train.src.3o_12138 > /home/yanym/save/enzh/src_infl/order/train.src.3o_token/train.src.3o_12138.BPE
"""

# python scripts/analysis/src_infl/out_of_order_token.py