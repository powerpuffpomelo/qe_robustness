# 统计train文件中每个token的出现次数，并写一个相同格式的次数文件存起来，其它文件以train文件中次数为准

# 根据qe_train 统计token与次数
qe_train_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt"

qe_train_cnt_dict = dict()
qe_train_cnt_dict_path = "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many/qe_train_token_cnt.dict"
with open(qe_train_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip('\n').split()
        for token in line:
            if token not in qe_train_cnt_dict:
                qe_train_cnt_dict[token] = 1
            else:
                qe_train_cnt_dict[token] += 1

data_list = ["/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt", 
                "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/dev.mt",
                "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt"]

save_list = ["/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many/qe_train_token_cnt",
                "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many/qe_dev_token_cnt",
                "/home/yanym/save/enzh/discriminator_finetuned_256_v4_worse_pseudo_mt_many/qe_test_token_cnt"]

for data_path, save_path in zip(data_list, save_list):
    with open(data_path, 'r', encoding='utf-8') as fd, open(save_path, 'w', encoding='utf-8') as fs:
        for line in fd.readlines():
            line = line.strip('\n').split()
            cnt_line = []
            for token in line:
                if token in qe_train_cnt_dict:
                    cnt_line.append(str(qe_train_cnt_dict[token]))
                else:
                    cnt_line.append('0')
            fs.write(' '.join(cnt_line) + '\n')

# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/analysis/training_data_analysis/token_cnt.py