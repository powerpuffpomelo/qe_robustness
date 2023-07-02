# 检查两个文件每行token数量是否一致
file1 = "/home/yanym/data/robust_word_alignment/train.src"
file2 = "/home/yanym/data/robust_word_alignment/train.pos"

with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
    for line1, line2 in zip(f1.readlines(), f2.readlines()):
        line1 = line1.strip('\n').split()
        line2 = line2.strip('\n').split()
        assert len(line1) == len(line2)

# python scripts/check_size.py