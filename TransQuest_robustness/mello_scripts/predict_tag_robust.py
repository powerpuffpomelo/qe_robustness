import os
from examples.word_level.common.util import reader, prepare_testdata
from examples.word_level.wmt_2020.en_zh.microtransquest_config_analysis import MODEL_TYPE, microtransquest_config
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

TEST_PATH = "/data1/yanym/data/robust_qe_test/generate_dev/"
TEST_SOURCE_FILE = "robust_test.src"
TEST_TARGET_FILE = "robust_test.mt"
TEST_MT_ROBUST_ID_FILE = "robust_test.idx"
TEMP_DIRECTORY = "/data1/yanym/data/robust_qe_test/pred_transquest"
TEST_SOURCE_TAGS_FILE = "robust_test.src_tag.pred"
TEST_TARGET_TAGS_FILE = "robust_test.mtgap_tag.pred"
TEST_MT_TAGS_FILE = "robust_test.mt_tag.pred"
TEST_GAP_TAGS_FILE = "robust_test.gap_tag.pred"
TEST_MT_ROBUST_TAGS_FILE = "robust_test.mt_tag_robust.pred"
"""
raw_test_df = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)
test_sentences = prepare_testdata(raw_test_df)

model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=["OK", "BAD"],
                            args=microtransquest_config)

sources_tags, targets_tags = model.predict(test_sentences, split_on_space=True)

with open(os.path.join(TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE), 'w', encoding='utf-8') as f:
    for src_tag_line in sources_tags:
        f.write(' '.join(src_tag_line) + '\n')

with open(os.path.join(TEMP_DIRECTORY, TEST_TARGET_TAGS_FILE), 'w', encoding='utf-8') as f, \
    open(os.path.join(TEMP_DIRECTORY, TEST_MT_TAGS_FILE), 'w', encoding='utf-8') as fm, \
    open(os.path.join(TEMP_DIRECTORY, TEST_GAP_TAGS_FILE), 'w', encoding='utf-8') as fg:
    for tgt_tag_line in targets_tags:
        f.write(' '.join(tgt_tag_line) + '\n')
        strm = ""
        strg = ""
        for i, tag in enumerate(tgt_tag_line):
            if i % 2 == 1:
                strm = strm + tag + ' '
            else:
                strg = strg + tag + ' '
        fm.write(strm.strip() + '\n')
        fg.write(strg.strip() + '\n')

"""
with open(os.path.join(TEMP_DIRECTORY, TEST_MT_TAGS_FILE), 'r', encoding='utf-8') as ft, \
    open(TEST_PATH + TEST_MT_ROBUST_ID_FILE, 'r', encoding='utf-8') as fid, \
    open(os.path.join(TEMP_DIRECTORY, TEST_MT_ROBUST_TAGS_FILE), 'w', encoding='utf-8') as fsave:
    all_tag_lines = ft.readlines()
    idx_lines = fid.readlines()
    for all_tag_line, idx_line in zip(all_tag_lines, idx_lines):
        all_tag_line = all_tag_line.strip('\n').split()
        idx_line = list(map(int, idx_line.strip('\n').split()))
        robust_tag_line = []
        for i, tag in enumerate(all_tag_line):
            if i in idx_line:
                robust_tag_line.append(tag)
        fsave.write(' '.join(robust_tag_line) + '\n')

"""
export CUDA_VISIBLE_DEVICES=7
python -m mello_scripts.predict_tag_robust
"""