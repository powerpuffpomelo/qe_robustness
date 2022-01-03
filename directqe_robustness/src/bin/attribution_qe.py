import argparse

from src.main import attribution_qe, attribution_qe_whole

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument("--pretrain_path", type=str,
                    help="""Path to model files.""")

parser.add_argument("--plot_path_prefix", type=str,
                    help="""Path to save plot.""")

parser.add_argument("--acc_sign_path", type=str, default="",
                    help="""Path to acc sign""")

parser.add_argument("--align_bpe_path_src2mt", type=str, default="",
                    help="""Path to align bpe file""")

parser.add_argument("--align_bpe_path_mt2src", type=str, default="",
                    help="""Path to align bpe file""")

parser.add_argument("--gradients_type", type=str,
                    help="""gradients_type: g, ig""")

parser.add_argument('--level', type=str, default=None, help='sentence level or word level')

parser.add_argument("--ig_steps", type=int, default=100,
                    help="""steps of ig""")

parser.add_argument("--test_id", type=int, default=0,
                    help="""id of focused test sample""")
                    
parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    attribution_qe(args)

if __name__ == '__main__':
    run()
