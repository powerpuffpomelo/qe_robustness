# 统计某token在训练集中出现次数、该token在训练集中的标准标注及预测标注、该token在训练集的预测准确率

from sklearn.metrics import accuracy_score

search_token = "它"

train_mt = "./train.mt"
train_tag_gold = "./train.tag"
train_tag_pred = "./train.tag.22800"

token_gold_list = []
token_pred_list = []

with open(train_mt, 'r', encoding='utf-8') as fmt, \
    open(train_tag_gold, 'r', encoding='utf-8') as fgold, \
    open(train_tag_pred, 'r', encoding='utf-8') as fpred:
    mt_lines = fmt.readlines()
    gold_lines = fgold.readlines()
    pred_lines = fpred.readlines()
    for mt_line, gold_line, pred_line in zip(mt_lines, gold_lines, pred_lines):
        mt_line = mt_line.strip('\n').split()
        gold_line = gold_line.strip('\n').split()
        pred_line = pred_line.strip('\n').split()
        for i, token in enumerate(mt_line):
            if token == search_token:
                token_gold_list.append(gold_line[i])
                token_pred_list.append(pred_line[i])
    
token_appear_num = len(token_gold_list)
if token_appear_num:
    token_gold_ok_num = sum([1 if tag == "OK" else 0 for tag in token_gold_list])
    token_pred_ok_num = sum([1 if tag == "OK" else 0 for tag in token_pred_list])
    token_gold_ok_ratio = token_gold_ok_num / token_appear_num
    token_pred_ok_ratio = token_pred_ok_num / token_appear_num
    acc = accuracy_score(token_gold_list, token_pred_list)

print("token_appear_num = %d" % token_appear_num)
print("token_gold_ok_ratio = %.4f" % token_gold_ok_ratio)
print("token_pred_ok_ratio = %.4f" % token_pred_ok_ratio)
print("pred_acc = %.4f" % acc)

# python token_stat.py