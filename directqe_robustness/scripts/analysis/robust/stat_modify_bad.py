# 观察从ok修改成bad的部分token，统计其改对的比例、以及在鲁棒性训练数据中出现过的比例
# 20.47
mt_bpe_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE"
gold_tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag.BPE"
pred1_tag_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec/test_result_10800/test.tag.BPE"    # 训练前预测结果
pred2_tag_path = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec_double/test_result_9600/test.tag.BPE"  # 训练后预测结果
robust_trained_set_path = "/data1/yanym/data/robust_qe_train_word2vec_double/robust_train.robust_token_set"

all_change_bad_num = 0
right_trained = 0
right_untrained = 0
wrong_trained = 0
wrong_untrained = 0

with open(mt_bpe_path, 'r', encoding='utf-8') as fmt, open(gold_tag_path, 'r', encoding='utf-8') as fgold, \
    open(pred1_tag_path, 'r', encoding='utf-8') as fpred1, open(pred2_tag_path, 'r', encoding='utf-8') as fpred2, \
    open(robust_trained_set_path, 'r', encoding='utf-8') as fset:

    trained_set = set()
    for set_line in fset.readlines():
        token = set_line.strip('\n')
        trained_set.add(token)

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
            if pred1_tag == "OK" and pred2_tag == "BAD":
                all_change_bad_num += 1
                if gold_tag == "BAD" and mt_token in trained_set: right_trained += 1
                elif gold_tag == "BAD" and mt_token not in trained_set: right_untrained += 1
                elif gold_tag == "OK" and mt_token in trained_set: wrong_trained += 1
                elif gold_tag == "OK" and mt_token not in trained_set: wrong_untrained += 1

print("all_change_bad_num: %d" % all_change_bad_num)
print("right_trained: %d" % right_trained)
print("right_untrained: %d" % right_untrained)
print("wrong_trained: %d" % wrong_trained)
print("wrong_untrained: %d" % wrong_untrained)
print("change_bad_change_right_ratio: %.4f" % ((right_trained + right_untrained) / all_change_bad_num))
print("change_bad_trained_ratio: %.4f" % ((right_trained + wrong_trained) / all_change_bad_num))


# python scripts/analysis/robust/stat_modify_bad.py