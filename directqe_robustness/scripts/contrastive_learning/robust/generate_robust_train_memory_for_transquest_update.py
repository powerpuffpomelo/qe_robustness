# 自动生成，基于训练记忆的数据增强鲁棒性训练集
"""
- 增强一批数据，遍历训练集中每个word，如果是大众word的话，就给它一条增强样本，这条增强样本只训练这一个word
  - 如果这个大众word是ok，就从错位文件中把这一条取出来；
  - 如果这个大众word是bad，就从小众样本列表里随机选一个，如果没有就不选
"""
import json
import random
from gensim.models.keyedvectors import KeyedVectors

random.seed(1234)

train_tag_stat_path = "/data1/yanym/data/robust_train_memory/train_token_tag_stat.json"
parallel_pos_stat_path = "/data1/yanym/data/robust_train_memory/parallel_word_position_stat.json"
with open(train_tag_stat_path, 'r', encoding='utf-8') as f1, open(parallel_pos_stat_path, 'r', encoding='utf-8') as f2:
    train_tag_stat = json.load(f1)
    parallel_pos_stat = json.load(f2)

word_vectors = KeyedVectors.load_word2vec_format(
    '/data1/yanym/save/word2vec/GoogleNews-vectors-negative300.bin', binary=True, limit=300000)

qe_train_src = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.src"
qe_train_mt = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.mt"
qe_train_src_tag = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.src_tag"
qe_train_mt_tag = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.mt_tag"
qe_train_mtgap_tag = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.mtgap_tag"
qe_train_hter = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/train.hter"
qe_train_align_mt2src = "/data1/yanym/data/robust_train_memory/align/train.mt-src.alignments.json"

parallel_src = "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/train.en_100w"
parallel_tgt = "/home/data_ti4_c/yanym/data/wmt20/enzh/parallel/mello_version/train.zh_100w"

robust_train_src = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.src"
robust_train_mt = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.mt"
robust_train_src_tag = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.src_tag"
robust_train_mt_tag = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.mt_tag"
robust_train_mtgap_tag = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.mtgap_tag"
robust_train_hter = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.hter"
robust_train_mt_idx_contrast = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.mt_idx_contrast"
robust_train_mt_idx_plus = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.mt_idx_plus"
robust_train_mtgap_idx = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.mtgap_idx"
robust_train_sample_seg = "/data1/yanym/data/robust_train_memory/data_augmentation_train_update_temp/robust_train.sample_seg.json"

f_ori_src = open(qe_train_src, 'r', encoding='utf-8')
f_ori_mt = open(qe_train_mt, 'r', encoding='utf-8')
f_ori_src_tag = open(qe_train_src_tag, 'r', encoding='utf-8')
f_ori_mt_tag = open(qe_train_mt_tag, 'r', encoding='utf-8')
f_ori_mtgap_tag = open(qe_train_mtgap_tag, 'r', encoding='utf-8')
f_ori_hter = open(qe_train_hter, 'r', encoding='utf-8')
f_ori_align = open(qe_train_align_mt2src, 'r', encoding='utf-8')

f_para_src = open(parallel_src, 'r', encoding='utf-8')
f_para_tgt = open(parallel_tgt, 'r', encoding='utf-8')

f_aug_src = open(robust_train_src, 'w', encoding='utf-8')
f_aug_mt = open(robust_train_mt, 'w', encoding='utf-8')
f_aug_src_tag = open(robust_train_src_tag, 'w', encoding='utf-8')
f_aug_mt_tag = open(robust_train_mt_tag, 'w', encoding='utf-8')
f_aug_mtgap_tag = open(robust_train_mtgap_tag, 'w', encoding='utf-8')
f_aug_hter = open(robust_train_hter, 'w', encoding='utf-8')
f_aug_mt_idx_contrast = open(robust_train_mt_idx_contrast, 'w', encoding='utf-8')
f_aug_mt_idx_plus = open(robust_train_mt_idx_plus, 'w', encoding='utf-8')
f_aug_mtgap_idx = open(robust_train_mtgap_idx, 'w', encoding='utf-8')
f_sample_seg = open(robust_train_sample_seg, 'w', encoding='utf-8')


ori_src_lines = f_ori_src.readlines()
ori_mt_lines = f_ori_mt.readlines()
ori_src_tag_lines = f_ori_src_tag.readlines()
ori_mt_tag_lines = f_ori_mt_tag.readlines()
ori_mtgap_tag_lines = f_ori_mtgap_tag.readlines()
ori_hter_lines = f_ori_hter.readlines()
align_dict_mt2src = json.load(f_ori_align)

robust_sample_seg = dict()

para_src_lines = f_para_src.readlines()
para_tgt_lines = f_para_tgt.readlines()

top_n_low = 3
top_n_up = 10
choose_k = 1

sent_id = 0
robust_sent_id = 0

for ori_src_line, ori_mt_line in zip(ori_src_lines, ori_mt_lines):
    #if sent_id > 3: break
    
    print(sent_id)
    robust_sample_seg[sent_id] = {'begin':0, 'end':0};
    robust_sample_seg[sent_id]['begin'] = robust_sent_id

    ori_src_tag_line = ori_src_tag_lines[sent_id]
    ori_mt_tag_line = ori_mt_tag_lines[sent_id]
    ori_mtgap_tag_line = ori_mtgap_tag_lines[sent_id]
    ori_hter_line = ori_hter_lines[sent_id]
    align_line = align_dict_mt2src[str(sent_id)]

    ori_src_line_list = ori_src_line.strip('\n').split()
    ori_src_tag_line_list = ori_src_tag_line.strip('\n').split()

    # 先抄一遍原始数据
    f_aug_src.write(ori_src_line)
    f_aug_mt.write(ori_mt_line)
    f_aug_src_tag.write(ori_src_tag_line)
    f_aug_mt_tag.write(ori_mt_tag_line)
    f_aug_mtgap_tag.write(ori_mtgap_tag_line)
    f_aug_hter.write(ori_hter_line)

    # 再做数据增强
    ori_mt_line_list = ori_mt_line.strip('\n').split()
    ori_mt_tag_line_list = ori_mt_tag_line.strip('\n').split()
    ori_mtgap_tag_line_list = ori_mtgap_tag_line.strip('\n').split()
    f_aug_mt_idx_plus.write(' '.join(map(str, list(range(len(ori_mt_line_list))))) + '\n')
    f_aug_mtgap_idx.write(' '.join(map(str, list(range(len(ori_mtgap_tag_line_list))))) + '\n')

    mt_idx_contrast_list_1 = []   # 对比idx部分预留，1为原始mt句idx，2为对比mt句idx，可能不是原句
    mt_idx_contrast_list_2 = []

    token_id = 0
    for token, tag in zip(ori_mt_line_list, ori_mt_tag_line_list):
        # 遍历每个词，如果一个词是大众词，则分情况讨论，okbad分别采用不同的数据增强方式
        if tag == 'OK' and train_tag_stat[token]['ok_ratio'] > 0.5: 
            if str(token_id) in align_line:   # 能在src找到对齐词汇
                align_position = align_line[str(token_id)]
                flag = True
                for pos in align_position:
                    if ori_src_line_list[pos] not in word_vectors:
                        flag = False
                        break
                if flag:
                    pseudo_mt_tag_list = ori_mt_tag_line_list[:]
                    pseudo_mt_tag_list[token_id] = "BAD"
                    pseudo_mtgap_tag_list = ori_mtgap_tag_line_list[:]
                    pseudo_mtgap_tag_list[token_id * 2 + 1] = "BAD"
                    pseudo_src_tag_list = ori_src_tag_line_list[:]
                    random_synonym_list = random.sample(range(top_n_low, top_n_up), choose_k)
                    for random_synonym in random_synonym_list:
                        for pos in align_position:
                            sim = word_vectors.most_similar(ori_src_line_list[pos], topn = top_n_up)
                            pseudo_src_list = ori_src_line_list[:]
                            pseudo_src_list[pos] = sim[random_synonym][0]
                            pseudo_src_tag_list[pos] = "BAD"
                    f_aug_src.write(' '.join(pseudo_src_list) + '\n')
                    f_aug_mt.write(ori_mt_line)
                    f_aug_src_tag.write(' '.join(pseudo_src_tag_list) + '\n')
                    f_aug_mt_tag.write(' '.join(pseudo_mt_tag_list) + '\n')
                    f_aug_mtgap_tag.write(' '.join(pseudo_mtgap_tag_list) + '\n')
                    f_aug_hter.write(ori_hter_line)
                    mt_idx_contrast_list_1.append(token_id)
                    mt_idx_contrast_list_2.append(token_id)
                    f_aug_mt_idx_plus.write(str(token_id) + '\n')
                    f_aug_mtgap_idx.write(str(token_id * 2 + 1) + '\n')
                    robust_sent_id += 1
        elif tag == 'BAD' and train_tag_stat[token]['ok_ratio'] < 0.5: 
            if train_tag_stat[token]['ok_freq'] != 0:
                random_niche = random.choice(train_tag_stat[token]['ok_pos'])
                f_aug_src.write(ori_src_lines[random_niche[0]])
                f_aug_mt.write(ori_mt_lines[random_niche[0]])
                f_aug_src_tag.write(ori_src_tag_lines[random_niche[0]])
                f_aug_mt_tag.write(ori_mt_tag_lines[random_niche[0]])
                f_aug_mtgap_tag.write(ori_mtgap_tag_lines[random_niche[0]])
                f_aug_hter.write(ori_hter_lines[random_niche[0]])
                mt_idx_contrast_list_1.append(token_id)
                mt_idx_contrast_list_2.append(random_niche[1])
                f_aug_mt_idx_plus.write(str(random_niche[1]) + '\n')
                f_aug_mtgap_idx.write(str(random_niche[1] * 2 + 1) + '\n')
                robust_sent_id += 1
            elif token in parallel_pos_stat:
                random_niche = random.choice(parallel_pos_stat[token]['pos_list'])
                src_line = para_src_lines[random_niche[0]]
                tgt_line = para_tgt_lines[random_niche[0]]
                src_line_list = src_line.strip('\n').split()
                tgt_line_list = tgt_line.strip('\n').split()
                f_aug_src.write(src_line)
                f_aug_mt.write(tgt_line)
                f_aug_src_tag.write(' '.join(['OK'] * len(src_line_list)) + '\n')
                f_aug_mt_tag.write(' '.join(['OK'] * len(tgt_line_list)) + '\n')
                f_aug_mtgap_tag.write(' '.join(['OK'] * (len(tgt_line_list) * 2 + 1)) + '\n')
                f_aug_hter.write('0.0\n')
                mt_idx_contrast_list_1.append(token_id)
                mt_idx_contrast_list_2.append(random_niche[1])
                f_aug_mt_idx_plus.write(str(random_niche[1]) + '\n')
                f_aug_mtgap_idx.write(str(random_niche[1] * 2 + 1) + '\n')
                robust_sent_id += 1
        
        token_id += 1

    # 一条样本遍历完，开始写contrast的idx
    f_aug_mt_idx_contrast.write(' '.join(map(str, mt_idx_contrast_list_1)) + '\n')
    for ii in mt_idx_contrast_list_2:
        f_aug_mt_idx_contrast.write(str(ii) + '\n')

    robust_sample_seg[sent_id]['end'] = robust_sent_id
    sent_id += 1
    robust_sent_id += 1

json.dump(robust_sample_seg, f_sample_seg, indent = 1)
            
# python scripts/contrastive_learning/robust/generate_robust_train_memory_for_transquest_update.py