import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math


hter_path = '/home/user_data_182b/yanym/qe/data/wmt19_ende/create_one_fault/test.hter'

stat = dict()
plt_stat = []
for i in range(10):
    stat[i] = 0

with open(hter_path, 'r', encoding='utf-8') as f_hter:
    for line in f_hter.readlines():
        hter = float(line.strip('\n'))
        plt_stat.append(hter)
        index = math.floor(hter * 10)
        if index == 10: index=9
        stat[index] += 1

print(stat)
# {0: 504, 1: 201, 2: 121, 3: 67, 4: 44, 5: 31, 6: 33, 7: 10, 8: 5, 9: 7}
plt.hist(plt_stat,bins=10, facecolor='g',alpha=0.75)
# plt.title('HTER分布')
plt.xlabel('HTER')
plt.ylabel('频数统计')
plt.grid(True)
plt.savefig('hist_test')

plt.hist(plt_stat, 1)

# python scripts/hter_statistics.py