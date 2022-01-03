# 统计一个标签预测文件里 ok bad 比例

file_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.gap_tag"
file_path = "/home/yanym/save/transquest/mt_infl_order/transquest_test_result_3o_sample/tag_src_77.pred"

all_num = 0
ok_num = 0
bad_num = 0
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split()
        num_list = [1 if tag == 'OK' else 0 for tag in line]   # 1ok  0bad
        this_all_num = len(num_list)
        this_ok_num = sum(num_list)
        this_bad_num = this_all_num - this_ok_num
        all_num += this_all_num
        ok_num += this_ok_num
        bad_num += this_bad_num

print("ok_ratio = %.4f" % (ok_num / all_num))
print("bad_ratio = %.4f" % (bad_num / all_num))

# python /home/data_ti4_c/yanym/code/TransQuest/mello_scripts/ok_ratio_stat.py