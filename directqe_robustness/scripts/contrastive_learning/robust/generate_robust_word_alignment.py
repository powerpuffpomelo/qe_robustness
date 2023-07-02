# 自动生成，基于 不能仅仅依赖词对齐的数据增强鲁棒性训练集
"""
对每条样本，遍历src每个word，找到其对齐的mt word的标签，
如果全是ok的话，就继续找下一个相同pos而且对齐不同mt的词，找得到就互换生成一条伪数据。
每条样本维护一个字典，键是pos，值是列表，列表中每个元素包括
 - 该src word位置id
 - 该src word
 - 该src word对应的mt word位置列表
 - 该src word对应的mt word是什么列表
 - 该src word对应的mt word标签列表
"""

add_origin_data = True  # 是否加入原始训练数据
is_tes_t = False   # 是测试场景，和是否加入原始数据相反

qe_train_src = "/data1/yanym/data/robust_word_alignment_wmt21/qe_data/train.src"
qe_train_mt = "/data1/yanym/data/robust_word_alignment_wmt21/qe_data/train.mt"
qe_train_tag = "/data1/yanym/data/robust_word_alignment_wmt21/qe_data/train.mt_tag"
qe_train_hter = "/data1/yanym/data/robust_word_alignment_wmt21/qe_data/train.hter"
qe_train_src_pos = "/data1/yanym/data/robust_word_alignment_wmt21/qe_data/train.src.pos"
qe_train_src_mt_align = "/data1/yanym/data/robust_word_alignment_wmt21/qe_data/train.src-mt.alignments"

robust_train_src = "/data1/yanym/data/robust_word_alignment_wmt21/robust_train.src"
robust_train_mt = "/data1/yanym/data/robust_word_alignment_wmt21/robust_train.mt"
robust_train_tag = "/data1/yanym/data/robust_word_alignment_wmt21/robust_train.mt_tag"
robust_train_hter = "/data1/yanym/data/robust_word_alignment_wmt21/robust_train.hter"
robust_train_idx = "/data1/yanym/data/robust_word_alignment_wmt21/robust_train.idx"

with open(qe_train_src, 'r', encoding='utf-8') as f_ori_src, open(qe_train_mt, 'r', encoding='utf-8') as f_ori_mt, \
    open(qe_train_tag, 'r', encoding='utf-8') as f_ori_tag, open(qe_train_hter, 'r', encoding='utf-8') as f_ori_hter, \
    open(qe_train_src_pos, 'r', encoding='utf-8') as f_ori_pos, open(qe_train_src_mt_align, 'r', encoding='utf-8') as f_ori_align, \
    open(robust_train_src, 'w', encoding='utf-8') as f_aug_src, open(robust_train_mt, 'w', encoding='utf-8') as f_aug_mt, \
    open(robust_train_tag, 'w', encoding='utf-8') as f_aug_tag, open(robust_train_hter, 'w', encoding='utf-8') as f_aug_hter, \
    open(robust_train_idx, 'w', encoding='utf-8') as f_aug_idx:
    ori_src_lines = f_ori_src.readlines()
    ori_mt_lines = f_ori_mt.readlines()
    ori_tag_lines = f_ori_tag.readlines()
    ori_hter_lines = f_ori_hter.readlines()
    ori_pos_lines = f_ori_pos.readlines()
    ori_align_lines = f_ori_align.readlines()

    sample_id = 0
    for ori_src_line, ori_mt_line, ori_tag_line, ori_hter_line, ori_pos_line, ori_align_line in zip(ori_src_lines, ori_mt_lines, ori_tag_lines, ori_hter_lines, ori_pos_lines, ori_align_lines):
        #if sample_id > 3: break

        ori_src_line_list = ori_src_line.strip('\n').split()
        ori_mt_line_list = ori_mt_line.strip('\n').split()
        ori_tag_line_list = ori_tag_line.strip('\n').split()
        ori_pos_line_list = ori_pos_line.strip('\n').split()
        ori_align_line_list = ori_align_line.strip('\n').split()
        
        align_dict = dict()
        for align_info in ori_align_line_list:
            src_id, mt_id = align_info.split('-')
            src_id = int(src_id)
            mt_id = int(mt_id)
            if src_id not in align_dict:
                align_dict[src_id] = []
            align_dict[src_id].append(mt_id)

        if add_origin_data:
            f_aug_src.write(ori_src_line)
            f_aug_mt.write(ori_mt_line)
            f_aug_tag.write(ori_tag_line)
            f_aug_hter.write(ori_hter_line)
            f_aug_idx.write(' '.join(map(str, list(range(len(ori_mt_line_list))))) + '\n')

        src_token_stat = dict()
        src_id = 0
        for src_token, src_token_pos in zip(ori_src_line_list, ori_pos_line_list):
            src_token_stat[src_id] = dict()
            src_token_stat[src_id]['src_id'] = src_id
            src_token_stat[src_id]['src_word'] = src_token
            src_token_stat[src_id]['src_pos'] = src_token_pos
            src_token_stat[src_id]['mt_id'] = []
            src_token_stat[src_id]['mt_word'] = []
            src_token_stat[src_id]['mt_tag'] = []
            if src_id in align_dict:
                for mt_id in align_dict[src_id]:
                    src_token_stat[src_id]['mt_id'].append(mt_id)
                    src_token_stat[src_id]['mt_word'].append(ori_mt_line_list[mt_id])
                    src_token_stat[src_id]['mt_tag'].append(ori_tag_line_list[mt_id])
            src_id += 1
        
        mt_id_all_list = []

        for src_id, src_token in enumerate(ori_src_line_list):
            mp = src_token_stat[src_id]
            choose = True
            if mp['mt_tag'] == []: 
                choose = False
            else:
                for mt_tag in mp['mt_tag']:
                    if mt_tag == 'BAD':
                        choose = False
                        break
            if choose == False: continue
            switch_src_id = -1
            for idx, ss in enumerate(ori_src_line_list):
                if idx == src_id: continue
                if src_token_stat[idx]['src_pos'] != mp['src_pos']: continue
                if list(set(src_token_stat[idx]['mt_id'])&set(mp['mt_id'])): continue
                switch_src_id = idx
                break
            if switch_src_id != -1:    # 找得到就互换
                switch_src_list = ori_src_line_list[:]
                switch_src_list[src_id], switch_src_list[switch_src_id] = switch_src_list[switch_src_id], switch_src_list[src_id]
                f_aug_src.write(' '.join(switch_src_list) + '\n')
                f_aug_mt.write(ori_mt_line)
                switch_tag_list = ori_tag_line_list[:]
                mt_id_all_list.extend(mp['mt_id'])
                for mt_id in mp['mt_id']:
                    switch_tag_list[mt_id] = "BAD"
                f_aug_tag.write(' '.join(switch_tag_list) + '\n')
                f_aug_hter.write(ori_hter_line)
                
                f_aug_idx.write(' '.join(list(map(str, mp['mt_id']))) + '\n')
            
        if is_tes_t and mt_id_all_list != []:
            f_aug_src.write(ori_src_line)
            f_aug_mt.write(ori_mt_line)
            f_aug_tag.write(ori_tag_line)
            f_aug_hter.write(ori_hter_line)
            f_aug_idx.write(' '.join(map(str, mt_id_all_list)) + '\n')

        
        sample_id += 1



# python scripts/contrastive_learning/robust/generate_robust_word_alignment.py