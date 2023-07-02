# 维护经过训练后修改正确的token集合，存起来

mt_bpe_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.mt.BPE"
gold_tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/test.tag.BPE"
pred1_tag_path = "/data1/yanym/save/temp/pretrain/test.tag.BPE"    # 训练前预测结果
pred2_tag_path = "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200_noshuffle/test.1200/test.tag.BPE"  # 训练后预测结果
modify_right_token_save_path = "/data1/yanym/save/directqe_enzh_bpe/discriminator_finetuned_256_v4_from11200_noshuffle/test.1200/modify_right_token_set"

with open(mt_bpe_path, 'r', encoding='utf-8') as fmt, open(gold_tag_path, 'r', encoding='utf-8') as fgold, \
    open(pred1_tag_path, 'r', encoding='utf-8') as fpred1, open(pred2_tag_path, 'r', encoding='utf-8') as fpred2, \
    open(modify_right_token_save_path, 'w', encoding='utf-8') as fset:
    modify_right_token_set = set()

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
            if pred2_tag == gold_tag and pred1_tag != gold_tag:
                modify_right_token_set.add(mt_token)

    modify_right_token_set = list(modify_right_token_set)
    for token in modify_right_token_set:
        fset.write(token + '\n')

# python scripts/analysis/robust/generate_modify_right_set.py