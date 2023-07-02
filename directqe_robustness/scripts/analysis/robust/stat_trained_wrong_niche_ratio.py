# 统计没训好的有多大比例是小众，小众有多大比例训不好

path_niche = "/data1/yanym/data/robust_boosting/train.niche_tag.BPE"
path_trained_wrong = "/data1/yanym/data/robust_boosting/train.trained_wrong_tag.BPE"

sum_n = 0
sum_w = 0
sum_n_w = 0  # 小众而且没训好的个数

sum_all = 0

with open(path_niche, 'r', encoding='utf-8') as fn, open(path_trained_wrong, 'r', encoding='utf-8') as fw:
    n_lines = fn.readlines()
    w_lines = fw.readlines()
    for n_line, w_line in zip(n_lines, w_lines):
        n_line = n_line.strip('\n').split()
        w_line = w_line.strip('\n').split()
        sum_all += len(n_line)
        for n_tag, w_tag in zip(n_line, w_line):
            if n_tag == '1':
                sum_n += 1
            if w_tag == '1':
                sum_w += 1
            if n_tag == '1' and w_tag == '1':
                sum_n_w += 1
    
print('全部token数量: %d' % sum_all)
print('小众token数量: %d' % sum_n)
print('小众token占比: %.4f' % (sum_n / sum_all))
print('没训好的token数量: %d' % sum_w)
print('没训好的token占比: %.4f' % (sum_w / sum_all))
print('没训好的token中的小众比例: %.4f' % (sum_n_w / sum_w))
print('小众token中没训好的比例: %.4f' % (sum_n_w / sum_n))

# python scripts/analysis/robust/stat_trained_wrong_niche_ratio.py