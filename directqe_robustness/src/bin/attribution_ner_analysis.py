import argparse

from src.main import attribution_ner_analysis

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--influence_path_prefix', type=str, default="/home/user_data_182b/yanym/qe/save/qe_analysis_influence/directqe_ende/",
                    help="The path for influence")

parser.add_argument('--ner_path_prefix', type=str, default="/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/NER/",
                    help="The path for ner")

parser.add_argument('--ner_lang', type=str, default="mt",
                    help="ner language to use: src or mt")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument("--recursion_depth", type=int, default=10,
                    help="""num of sampling""")
                    
parser.add_argument("--num_first", type=int, default=10,
                    help="""num of most influential train samples focused""")

parser.add_argument("--steps", type=int, default=100,
                    help="""steps of ig""")

parser.add_argument("--test_id", type=int, default=0,
                    help="""id of focused test sample""")

parser.add_argument("--test_token", type=int, default=0,
                    help="""id of focused token of focused test sample""")

parser.add_argument("--pretrain_path", type=str,
                    help="""Path to model files.""")

parser.add_argument("--gradients_type", type=str,
                    help="""gradients_type: ig, gi""")

parser.add_argument("--saliency_level", type=str, default="hter",
                    help="""hter or token, which level of saliency""")

parser.add_argument("--saliency_token", type=int, default=-1,
                    help="""id of focused test token to [which token], if -1, it's the same as focused test token""")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    attribution_ner_analysis(args)

if __name__ == '__main__':
    run()
