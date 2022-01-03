import random
seed = 12138
random.seed(seed)

file_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.mt"
file_3o = "/home/yanym/save/transquest/mt_infl_order/train.mt.3o_sample/train.mt.3o_" + str(seed)

with open(file_path, 'r', encoding='utf-8') as f1, open(file_3o, 'w', encoding='utf-8') as f2:
    src_lines = f1.readlines()
    random.shuffle(src_lines)
    for line in src_lines:
        f2.write(line)


"""
python scripts/analysis/src_infl/out_of_order_sample.py
subword-nmt apply-bpe -c /home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/bpe.30000.en < /home/yanym/save/enzh/src_infl/order/train.src.3o_sample/train.src.3o_77 > /home/yanym/save/enzh/src_infl/order/train.src.3o_sample/train.src.3o_77.BPE

"""
