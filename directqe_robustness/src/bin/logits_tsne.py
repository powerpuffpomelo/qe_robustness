import argparse

from src.main import logits_tsne

parser = argparse.ArgumentParser()

parser.add_argument("--pretrain_path_1", type=str,
                    help="""Path to model files.""")

parser.add_argument("--pretrain_path_2", type=str,
                    help="""Path to model files.""")

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    logits_tsne(args)

if __name__ == '__main__':
    run()
