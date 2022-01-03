import torch

influence_path_prefix = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence/directqe_ende_gi_no_weight/"

for f_str in ['0_1', '1_0', '2_3', '3_4']:
    infl_path = influence_path_prefix + 'test_' + f_str + '.train_infl'
    train_influences = torch.load(infl_path)
    infl_sort_id = sorted(enumerate(train_influences), key=lambda x:x[1], reverse=True)
    infl_sort_id = [x[0] for x in infl_sort_id]

    log_path = influence_path + 'log_' + f_str + '.train_infl'
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(' '.join([str(id) for id in infl_sort_id]) + '\n\n')
        for id in infl_sort_id:
            infl_sample = train_dataset.showitem(id)
            logf.write('---------------------------train id: ' + str(id) + '---------------------------\n')
            for sample_i in infl_sample:
                logf.write(sample_i)

# 还是借用别的函数方便