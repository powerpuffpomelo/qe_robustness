import json

vocab_u = "/home/yanym/save/data/vocab_2kw_7k_100.zh"
vocab_s = "/home/yanym/save/data/vocab_2kw_7k_100.zh.new"

with open(vocab_u, 'r', encoding='utf-8') as fu, open(vocab_s, 'w', encoding='utf-8') as fs:
    vocab = json.load(fu)
    fs.write((json.dumps(vocab, indent = 1)).encode('utf-8').decode("unicode-escape"))
    


# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/change_vocab_code.py