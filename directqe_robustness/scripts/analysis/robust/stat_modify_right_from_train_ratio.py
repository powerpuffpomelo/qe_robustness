modify_right_set_path = "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200_noshuffle/test.1200/modify_right_token_set"
train_set_path = "/data1/yanym/data/normal_train/normal_train.token_set"

all_num = 0
trained_num = 0

with open(modify_right_set_path, 'r', encoding='utf-8') as fmo, open(train_set_path, 'r', encoding='utf-8') as ftrain:
    train_set = set()
    for train_token_line in ftrain.readlines():
        train_token = train_token_line.strip('\n')
        train_set.add(train_token)
    for modify_right_token_line in fmo.readlines():
        modify_right_token = modify_right_token_line.strip('\n')
        if modify_right_token in train_set: trained_num += 1
        all_num += 1

print("all_num: %.4f" % all_num)
print("trained_num: %.4f" % trained_num)
print("ratio of modify_right_token from train_set: %.4f" % (trained_num / all_num))

# python scripts/analysis/robust/stat_modify_right_from_train_ratio.py