import torch
from scipy.stats import pearsonr

influence_path = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence/directqe_ende/"

for f_str in ['0_1', '1_0', '2_3', '3_4']:
    f1 = influence_path + 'test_' + f_str + '.train_infl'
    f2 = influence_path + 'test_' + f_str + '.train_infl_before'
    infl_1 = torch.load(f1)
    infl_2 = torch.load(f2)
    infl_sort_id1 = sorted(enumerate(infl_1), key=lambda x:x[1], reverse=True)
    infl_sort_id1 = [x[0] for x in infl_sort_id1]
    infl_sort_id2 = sorted(enumerate(infl_2), key=lambda x:x[1], reverse=True)
    infl_sort_id2 = [x[0] for x in infl_sort_id2]
    print(f_str)
    print ("pearson_corrcoef_20 = %.4f" % pearsonr(infl_sort_id1[:20], infl_sort_id2[:20])[0])
    print ("pearson_corrcoef_all = %.4f" % pearsonr(infl_sort_id1, infl_sort_id2)[0])


# python scripts/attribution/cal_p_for_sample_size.py