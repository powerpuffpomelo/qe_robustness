# 用bad比例估算hter
tag_file = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v2/test_result/test.tag"
simulate_hter_file = "/home/yanym/save/enzh/vocab2/test.hter_simulate"
with open(tag_file, 'r', encoding='utf-8') as f_tag, open(simulate_hter_file, 'w', encoding='utf-8') as f_hter:
    lines = f_tag.readlines()
    for line in lines:
        line = line.strip('\n').split()
        bad_list = [1 if token == 'BAD' else 0 for token in line]
        f_hter.write("%.6f\n" % (sum(bad_list) / len(bad_list)))
        #f_hter.write(str(sum(bad_list)) + ' ' + str(len(bad_list)) + '\n')

# python scripts/analysis/statistics/simulate_hter.py