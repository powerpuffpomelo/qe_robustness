gold_tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_test/test.gap_tag"
pred_tag_path = "/data1/yanym/save/transquest_wmt20_enzh/test_result/test.gap_tag.pred"
tag_acc_sign_path = "/data1/yanym/test.gap_tag.acc_sign"

with open(gold_tag_path, 'r', encoding='utf-8') as fg, open(pred_tag_path, 'r', encoding='utf-8') as fp, \
    open(tag_acc_sign_path, 'w', encoding='utf-8') as fs:
    gold_tag_lines = fg.readlines()
    pred_tag_lines = fp.readlines()
    for gold_line, pred_line in zip(gold_tag_lines, pred_tag_lines):
        gold_line = gold_line.strip('\n').split()
        pred_line = pred_line.strip('\n').split()
        tag_acc_sign_list = ['1' if gold_line[i] == pred_line[i] else '0' for i in range(len(gold_line))]
        fs.write(' '.join(tag_acc_sign_list) + '\n')

# python mello_scripts/tag_acc_sign.py