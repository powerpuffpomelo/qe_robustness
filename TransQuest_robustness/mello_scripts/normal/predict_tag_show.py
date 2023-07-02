import os
from examples.word_level.common.util import reader, prepare_testdata
from examples.word_level.wmt_2020.en_zh.microtransquest_config_analysis import MODEL_TYPE, microtransquest_config
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

TEST_PATH = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_test/"
TEST_SOURCE_FILE = "test.src"
TEST_TARGET_FILE = "test.mt"

#raw_test_df = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)
#test_sentences = prepare_testdata(raw_test_df)[0:1]
origin_test_sentences = [['rowboats and pedalboats can be rented at the boathouse .', '游艇 和 脚踏船 可以 在 船坞 租用 .']]
modified_test_sentences = [['rowboats and pedalboats can be abandoned at the boathouse .', '游艇 和 脚踏船 可以 在 船坞 租用 .']]

model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=["OK", "BAD"], use_cuda=False,
                            args=microtransquest_config)

origin_sources_tags, origin_targets_tags = model.predict(origin_test_sentences, split_on_space=True)
modified_sources_tags, modified_targets_tags = model.predict(modified_test_sentences, split_on_space=True)

print("===================== origin =====================")
print(origin_test_sentences)
print(origin_sources_tags)
print(origin_targets_tags)
print("===================== modified =====================")
print(modified_test_sentences)
print(modified_sources_tags)
print(modified_targets_tags)

"""
export CUDA_VISIBLE_DEVICES=3
python -m mello_scripts.predict_tag_show
"""