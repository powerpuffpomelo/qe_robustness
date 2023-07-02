import stanza
import json


# Download an English model into the default directory
#print("Downloading English model...")
#stanza.download('en')

# Similarly, download a (simplified) Chinese model
# Note that you can use verbose=False to turn off all printed messages
# print("Downloading Chinese model...")
#stanza.download('de', verbose=False)


# Build an English pipeline, with all processors by default
#print("Building an English pipeline...")
#en_nlp = stanza.Pipeline('en')

# Build a Chinese pipeline, with customized processor list and no logging, and force it to use CPU
# print("Building a Chinese pipeline...")
# zh_nlp = stanza.Pipeline('zh', processors='tokenize,lemma,pos,depparse', verbose=False, tokenize_pretokenized=True)
en_nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos,depparse', verbose=False, tokenize_pretokenized=True)
path_prefix = '/home/yanym/qe_data/'

for split in ['train', 'test']:
    with open(path_prefix+split+'.src', 'r', encoding='utf-8') as f, open(path_prefix+split+'.src.pos', 'w', encoding='utf-8') as fpos:
        for i,line in enumerate(f.readlines()):
            doc = en_nlp(line)
            for sent in doc.sentences:
                poss = []
                for word in sent.words:
                    poss.append(word.pos)
                pos_line = " ".join(poss)+'\n'
                fpos.write(pos_line)

"""
#print("Downloading English model...")
#stanza.download('en')

#print("Downloading Chinese model...")
#stanza.download('zh')

#print("Downloading Germany model...")
#stanza.download('de')


# 得到src英文的ner
en_nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', verbose=False, tokenize_pretokenized=True)
path_prefix = '/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/'

for split in ['train.src', 'dev.src', 'test.src']:
    with open(path_prefix+split, 'r', encoding='utf-8') as f, open(path_prefix+split+'.ner.json', 'w', encoding='utf-8') as fner:
        ner_dict = {}
        for i,line in enumerate(f.readlines()):
            doc = en_nlp(line)
            for sent in doc.sentences:
                # ner = [(ent.type, ent.text) for ent in sent.ents]
                ner = [ent.type for ent in sent.ents]
            ner_dict[i] = ner      # 每个句子维护一个ner列表
        json.dump(ner_dict, fner, indent=1)



# 得到mt中文的ner
zh_nlp = stanza.Pipeline(lang='zh', processors='tokenize,ner', verbose=False, tokenize_pretokenized=True)

for split in ['train.mt', 'dev.mt', 'test.mt']:
    with open(path_prefix+split, 'r', encoding='utf-8') as f, open(path_prefix+split+'.ner.json', 'w', encoding='utf-8') as fner:
        ner_dict = {}
        for i,line in enumerate(f.readlines()):
            doc = zh_nlp(line)
            for sent in doc.sentences:
                # ner = [(ent.type, ent.text) for ent in sent.ents]
                ner = [ent.type for ent in sent.ents]
            ner_dict[i] = ner
        json.dump(ner_dict, fner, indent=1)
"""
        
# TODO: 怎么指定GPU？
# python ./scripts/use_stanza.py