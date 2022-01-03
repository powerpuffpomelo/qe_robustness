# 从test集中把带某个token的样本都抽出来，形成新的token_test集

extract_token = "adopt"

test_src = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.src.BPE"
test_mt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt"
test_tag_gold = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag"
test_tag_pred = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/test_result/test.tag"

test_src = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.src.BPE"
test_mt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt"
test_tag_gold = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.tag"
test_tag_pred = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v4_worse_pseudo_mt_sent/test_result/train.tag"

save_prefix = "/home/yanym/save/enzh/src_infl/extract_token_sample/"
token_test_src = save_prefix + extract_token + ".src.BPE"
token_test_mt = save_prefix + extract_token + ".mt"
token_test_tag_gold = save_prefix + extract_token + ".tag_gold"
token_test_tag_pred = save_prefix + extract_token + ".tag_pred"

with open(test_src, 'r', encoding='utf-8') as fs, open(test_mt, 'r', encoding='utf-8') as fm, \
    open(test_tag_gold, 'r', encoding='utf-8') as fg, open(test_tag_pred, 'r', encoding='utf-8') as fp, \
    open(token_test_src, 'w', encoding='utf-8') as fss, open(token_test_mt, 'w', encoding='utf-8') as fmm, \
    open(token_test_tag_gold, 'w', encoding='utf-8') as fgg, open(token_test_tag_pred, 'w', encoding='utf-8') as fpp:
    test_src_lines = fs.readlines()
    test_mt_lines = fm.readlines()
    test_gold_lines = fg.readlines()
    test_pred_lines = fp.readlines()
    for src_line, mt_line, gold_line, pred_line in zip(test_src_lines, test_mt_lines, test_gold_lines, test_pred_lines):
        src_tokens = src_line.strip('\n').split()
        mt_tokens = mt_line.strip('\n').split()
        if extract_token in src_tokens:
            fss.write(src_line)
            fmm.write(mt_line)
            fgg.write(gold_line)
            fpp.write(pred_line)

# python /home/data_ti4_c/yanym/code/directqe_analysis/scripts/analysis/src_infl/extract_token_sample.py