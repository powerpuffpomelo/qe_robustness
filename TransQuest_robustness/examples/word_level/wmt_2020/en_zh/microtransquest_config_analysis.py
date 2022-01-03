from multiprocessing import cpu_count

TRAIN_PATH = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_train/"
TRAIN_SOURCE_FILE = "train.src"
TRAIN_SOURCE_TAGS_FILE = "train.src_tag"
TRAIN_TARGET_FILE = "train.mt"
TRAIN_TARGET_TAGS_FLE = "train.mtgap_tag"

DEV_PATH = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_dev/"
DEV_SOURCE_FILE = "dev.src"
DEV_SOURCE_TAGS_FILE = "dev.src_tag"
DEV_TARGET_FILE = "dev.mt"
DEV_TARGET_TAGS_FLE = "dev.mtgap_tag"

TEST_PATH = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_test/"
TEST_SOURCE_FILE = "test.src"
TEST_TARGET_FILE = "test.mt"

TEST_SOURCE_TAGS_FILE = "test.src_tag.pred"
TEST_TARGET_TAGS_FLE = "test.mtgap_tag.pred"

DEV_SOURCE_TAGS_FILE_SUB = "dev.src_tag.pred"
DEV_TARGET_TAGS_FILE_SUB = "dev.mtgap_tag.pred"

plot_path_prefix = "/data1/yanym/save/transquest_wmt20_enzh/grad_plot/"
test_align_file_path = "/home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_transquest/qe_test/test.src-mt.alignments"
test_src_tag_acc_sign_path = "/data1/yanym/save/transquest_wmt20_enzh/test_result/test.src_tag.acc_sign"
test_mt_tag_acc_sign_path = "/data1/yanym/save/transquest_wmt20_enzh/test_result/test.mt_tag.acc_sign"
test_gap_tag_acc_sign_path = "/data1/yanym/save/transquest_wmt20_enzh/test_result/test.gap_tag.acc_sign"

k_adj = 1

SEED = 777
TEMP_DIRECTORY = "/data1/yanym/save/transquest_wmt20_enzh/data"
GOOGLE_DRIVE = False
DRIVE_FILE_ID = None
MODEL_TYPE = "xlmroberta"
MODEL_NAME = "xlm-roberta-large"

microtransquest_config = {
    'output_dir': '/data1/yanym/save/transquest_wmt20_enzh/outputs/',
    "best_model_dir": "/data1/yanym/save/transquest_wmt20_enzh/outputs/best_model",
    'cache_dir': '/data1/yanym/save/transquest_wmt20_enzh/cache_dir/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 200,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 1,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 300,
    'save_steps': 300,
    "no_cache": False,
    "no_save": False,
    "save_recent_only": True,
    'save_model_every_epoch': False,
    'n_fold': 3,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": True,
    'evaluate_during_training_steps': 300,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': True,
    'tensorboard_dir': '/data1/yanym/save/transquest_wmt20_enzh/runs/test_run',
    "save_optimizer_and_scheduler": True,

    'regression': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    "multiprocessing_chunksize": 500,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    "manual_seed": SEED,

    "add_tag": False,
    "tag": "_",

    "default_quality": "OK",

    "config": {},
    "local_rank": -1,
    "encoding": None,

    "source_column": "source",
    "target_column": "target",
    "source_tags_column": "source_tags",
    "target_tags_column": "target_tags",
}
