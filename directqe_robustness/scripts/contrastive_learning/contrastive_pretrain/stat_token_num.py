tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag.BPE"
num = 0

with open(tag_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split()
        num += len(line)

print(num)

# python scripts/contrastive_learning/contrastive_pretrain/stat_token_num.py