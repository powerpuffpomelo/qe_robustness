# 统计某一个token在数据集中的标注ok bad比例

mt_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.mt"
tag_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.tag"
pred_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version/train.tag_pred"

tt = '狼'
tt_ok_count = 0
tt_bad_count = 0
tt_ratio = 0
pred_ok_count = 0
pred_bad_count = 0
pred_ratio = 0

with open(mt_path, 'r', encoding='utf-8') as f_mt, open(tag_path, 'r', encoding='utf-8') as f_tag, \
    open(pred_path, 'r', encoding='utf-8') as f_pred:
    lines_mt = f_mt.readlines()
    lines_tag = f_tag.readlines()
    lines_pred = f_pred.readlines()
    id_sent = 0
    for line_mt, line_tag, line_pred in zip(lines_mt, lines_tag, lines_pred):
        #print(id_sent)
        line_mt = line_mt.strip('\n').split()
        line_tag = line_tag.strip('\n').split()
        line_pred = line_pred.strip('\n').split()
        assert len(line_mt)==len(line_tag)
        
        if tt in line_mt:
            id = line_mt.index(tt)
            #print(id_sent)
            #print(id)
            tt_tag = line_tag[id]
            pred_tag = line_pred[id]

            if tt_tag == 'BAD': tt_bad_count += 1
            else: tt_ok_count += 1
            if pred_tag == 'BAD': pred_bad_count += 1
            else: pred_ok_count += 1
            #print(tt_tag)
            #assert 1==2
        id_sent += 1
    
    print(tt_ok_count)
    print(tt_bad_count)
    print(pred_ok_count)
    print(pred_bad_count)

    tt_ratio = tt_ok_count / tt_bad_count
    pred_ratio = pred_ok_count / pred_bad_count
    
    print(tt_ratio)
    print(pred_ratio)





# python scripts/analysis/statistics/token_tag_ratio.py