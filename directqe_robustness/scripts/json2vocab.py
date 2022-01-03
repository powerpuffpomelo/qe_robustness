import json

vocab_json = "/home/yanym/save/data/vocab_2kw_7k_100.zh.u"
vocab_file = "/home/yanym/save/data/vocab_2kw_7k_100.zh"


with open(vocab_json, 'r', encoding='utf-8') as fjson, open(vocab_file, 'w', encoding='utf-8') as fvocab:
    vocab = json.load(fjson)
    for word, value in vocab.items():
        num = str(value[-1])
        line = word+' '+num+'\n'
        fvocab.write(line)

# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/json2vocab.py