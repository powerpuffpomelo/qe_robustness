hter_file = "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/wmt20_enzh_pseudo/pseudo_100w.hter_pred"

with open(hter_file, 'r', encoding='utf-8') as f:
    sum = 0
    lines = f.readlines()
    num = len(lines)
    for line in lines:
        hter = float(line.strip('\n'))
        sum += hter


avg = sum / num
print("%.6f" % avg)

# python scripts/train_directqe_enzh/analysis/cal_avg.py