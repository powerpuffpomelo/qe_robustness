import os
from examples.word_level.common.util import reader, prepare_testdata
from examples.word_level.wmt_2020.en_de.microtransquest_config_analysis import MODEL_TYPE, microtransquest_config
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

TEST_PATH = "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version_transquest/qe_test/"
TEST_SOURCE_FILE = "test.src"
TEST_TARGET_FILE = "test.mt"
TEMP_DIRECTORY = "/data1/yanym/save/transquest_wmt20_ende/test_result"
TEST_SOURCE_TAGS_FILE = "test.src_tag.pred"
TEST_TARGET_TAGS_FILE = "test.mtgap_tag.pred"
TEST_MT_TAGS_FILE = "test.mt_tag.pred"
TEST_GAP_TAGS_FILE = "test.gap_tag.pred"

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
export CUDA_VISIBLE_DEVICES=7
python -m mello_scripts.predict_tag_ende
"""
# 就一定要 python -m mello_scripts.predict_tag，不然 python mello_scripts/predict_tag.py 就会报错找不到examples包