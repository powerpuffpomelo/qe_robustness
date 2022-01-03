import os
from examples.word_level.common.util import reader, prepare_testdata
from examples.word_level.wmt_2020.en_zh.microtransquest_config_analysis import MODEL_TYPE, microtransquest_config, plot_path_prefix, test_align_file_path, k_adj, \
    test_src_tag_acc_sign_path, test_mt_tag_acc_sign_path, test_gap_tag_acc_sign_path
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

TEST_PATH = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_test/"
TEST_SOURCE_FILE = "test.src"
TEST_TARGET_FILE = "test.mt"

raw_test_df = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)
test_sentences = prepare_testdata(raw_test_df)

model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=["OK", "BAD"], use_cuda=True,
                            args=microtransquest_config)

model.attribution(test_sentences, plot_path_prefix=plot_path_prefix, align_file_path=test_align_file_path, \
        src_tag_acc_sign_path=test_src_tag_acc_sign_path, mt_tag_acc_sign_path=test_mt_tag_acc_sign_path, \
        gap_tag_acc_sign_path=test_gap_tag_acc_sign_path, k_adj=k_adj, split_on_space=True)



"""
export CUDA_VISIBLE_DEVICES=6
echo "Using GPU $CUDA_VISIBLE_DEVICES..."
python -m mello_scripts.attribution_enzh
"""