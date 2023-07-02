# 观察从ok修改成bad的部分token，统计其改对的比例、以及在鲁棒性训练数据中出现过的比例
import json

mt_bpe_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE"
gold_tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag.BPE"
pred1_tag_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec/test_result_10800/test.tag.BPE"    # 训练前预测结果
pred2_tag_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/test_result_9600/test.tag.BPE"  # 训练后预测结果
robust_trained_set_path = "/data1/yanym/data/robust_qe_train_word2vec_double/robust_train.robust_token_set"
train_stat_dict_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/test_result_9600/robust_train_word2vec_double_stat.json"
test_stat_dict_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/test_result_9600/robust_test_word2vec_limit_3_10_stat.json"

trained_change_bad_num = 0
change_right_num = 0
change_wrong_num = 0
train_acc_sum_right = 0
train_acc_sum_wrong = 0
test_acc_sum_right = 0
test_acc_sum_wrong = 0

with open(mt_bpe_path, 'r', encoding='utf-8') as fmt, open(gold_tag_path, 'r', encoding='utf-8') as fgold, \
    open(pred1_tag_path, 'r', encoding='utf-8') as fpred1, open(pred2_tag_path, 'r', encoding='utf-8') as fpred2, \
    open(train_stat_dict_path, 'r', encoding='utf-8') as ftrain, open(test_stat_dict_path, 'r', encoding='utf-8') as ftest:

    train_stat = json.load(ftrain)
    test_stat = json.load(ftest)

    mt_lines = fmt.readlines()
    gold_tag_lines = fgold.readlines()
    pred1_tag_lines = fpred1.readlines()
    pred2_tag_lines = fpred2.readlines()
    for mt_line, gold_tag_line, pred1_tag_line, pred2_tag_line in zip(mt_lines, gold_tag_lines, pred1_tag_lines, pred2_tag_lines):
        mt_line = mt_line.strip('\n').split()
        gold_tag_line = gold_tag_line.strip('\n').split()
        pred1_tag_line = pred1_tag_line.strip('\n').split()
        pred2_tag_line = pred2_tag_line.strip('\n').split()
        for mt_token, gold_tag, pred1_tag, pred2_tag in zip(mt_line, gold_tag_line, pred1_tag_line, pred2_tag_line):
            if pred1_tag == "OK" and pred2_tag == "BAD" and mt_token in train_stat and mt_token in test_stat:
                trained_change_bad_num += 1
                if gold_tag == "BAD":
                    change_right_num += 1
                    train_acc_sum_right += train_stat[mt_token]['acc']
                    test_acc_sum_right += test_stat[mt_token]['acc']
                else:
                    change_wrong_num += 1
                    train_acc_sum_wrong += train_stat[mt_token]['acc']
                    test_acc_sum_wrong += test_stat[mt_token]['acc']


print("trained_change_bad_num: %d" % trained_change_bad_num)
print("change_right_ratio: %.4f" % (change_right_num / trained_change_bad_num))
print("change_right_train_avg_acc: %.4f" % (train_acc_sum_right / change_right_num))
print("change_wrong_train_avg_acc: %.4f" % (train_acc_sum_wrong / change_wrong_num))
print("change_right_test_avg_acc: %.4f" % (test_acc_sum_right / change_right_num))
print("change_wrong_test_avg_acc: %.4f" % (test_acc_sum_wrong / change_wrong_num))

# python scripts/analysis/robust/stat_modify_bad_new.py