file_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/parallel_plus_pseudo_from_src/pseudo_train.mt.BPE"
index_path = "/data1/yanym/data/alignments/parallel_train.index"

def get_id(file_path, index_path):
    with open(file_path, 'r', encoding='utf-8') as f, open(index_path, 'w', encoding='utf-8') as fid:
        for id, line in enumerate(f.readlines()):
            fid.write(str(id) + '\n')

get_id(file_path, index_path)

# python scripts/contrastive_learning/contrastive_pretrain/get_index.py