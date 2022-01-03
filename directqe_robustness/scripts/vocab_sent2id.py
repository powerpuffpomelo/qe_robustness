
vocab_file = "/home/yanym/save/data/vocab_2kw_7k_100.zh"
test_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt"

with open(vocab_file, 'r', encoding='utf-8') as fv, open(test_file, 'r', encoding='utf-8') as ft:
    vocab = dict()
    for i, line in enumerate(fv):
        ww = line.strip().split()[0]
        vocab[ww] = (i, 0)
    aa = 0
    for line in ft.readlines():
        line = line.strip('\n').split()
        line = [vocab[w][0] if w in vocab else -1 for w in line]
        print(line)
        aa += 1
        if aa == 4: break
    
    


# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/vocab_sent2id.py