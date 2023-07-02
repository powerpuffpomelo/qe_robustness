from scipy.stats import pearsonr

def pearson_corrcoef(x, y):
    return pearsonr(x, y)[0]

def read_file(filePath):
    fr = open(filePath, 'r')
    lines = fr.readlines()
    dataList = []
    for line in lines:
        line = line.strip('\n').split()
        line = map(int, line)
        dataList.extend(line)
    return dataList

path1 = "/data1/yanym/data/robust_boosting/train.niche_tag.BPE"
path2 = "/data1/yanym/data/robust_boosting/train.trained_wrong_tag.BPE"

list1 = read_file(path1)
list2 = read_file(path2)

print ("pearson_corrcoef = %.4f" % pearson_corrcoef(list1, list2))

# python scripts/contrastive_learning/robust/cal_pearsonr.py