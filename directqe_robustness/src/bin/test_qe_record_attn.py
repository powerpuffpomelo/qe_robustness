import argparse

from src.main import test_qe_record_attn

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--saveto_tags', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--saveto_hter', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--saveto_attn', type=str, default="./save",
                    help="The path for saving attn. Default is ./save.")

parser.add_argument('--plot_path_prefix', type=str, default="./save",
                    help="The path for saving plot. Default is ./save.")

parser.add_argument('--saveto_okprob', type=str, default=None,
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--add_align', action="store_true",
                    help="Whether to use adding method instead of attn modification.")

parser.add_argument("--source_path", type=str,
                    help="""Path to source file.""")

parser.add_argument("--target_path", type=str,
                    help="""Path to target file.""")

parser.add_argument("--align_bpe_path_mt2src_test", type=str, default=None,
                    help="""Path to align bpe file""")

parser.add_argument("--align_ratio", type=float, default=0,
                    help="""ratio of align_matrix.   1-align_ratio = ratio of attn_matrix""")

parser.add_argument('--level', type=str, default=None, help='sentence level or word level')

parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument("--pretrain_path", type=str,
                    help="""Path to model files.""")

def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    test_qe_record_attn(args)

if __name__ == '__main__':
    run()
