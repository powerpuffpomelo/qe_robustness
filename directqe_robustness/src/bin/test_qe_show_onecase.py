import argparse

from src.main import test_qe_show_onecase

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument("--src_sent", type=str,
                    help="""src sent""")

parser.add_argument("--mt_sent", type=str,
                    help="""mt sent""")

parser.add_argument("--batch_size", type=int, default=1,
                    help="""Batch size of beam search.""")

parser.add_argument("--pretrain_path", type=str,
                    help="""Path to model files.""")

def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    test_qe_show_onecase(args)

if __name__ == '__main__':
    run()
