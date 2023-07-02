# 统计qe数据集中的ok bad比例


#qe_train_tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.tag"
qe_train_tag_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/test_result/robust_test_word2vec_limit_3_10.tag_idxonly.BPE"
#qe_train_hter_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/dev.hter"
#qe_train_hter_path = "/home/yanym/save/enzh/generate_dev/mlm_dev.hter"

num_ok = 0
num_bad = 0
sum_hter = 0


with open(qe_train_tag_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split()
        line_int = [1 if item == 'OK' else 0 for item in line]
        okk = sum(line_int)
        num_ok += okk
        num_bad += len(line) - okk

"""
with open(qe_train_hter_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        sum_hter += float(line.strip('\n'))
    print('mean_hter = %f' % (sum_hter / len(lines)))
"""  

print(num_ok)
print(num_bad)
print("ok ratio: %.4f" % (num_ok / (num_ok + num_bad)))



# python scripts/analysis/statistics/data_bad_ratio.py