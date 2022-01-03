import argparse

from src.main import test_qe_knn_token

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--saveto_tags', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--saveto_hter', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--saveto_okprob', type=str, default=None,
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--faiss_index_prefix', type=str, default="/home/yanym/save/enzh/faiss_index/",
                    help="The path for faiss_index_prefix")

parser.add_argument("--source_path", type=str,
                    help="""Path to source file.""")

parser.add_argument("--target_path", type=str,
                    help="""Path to target file.""")

parser.add_argument('--level', type=str, default=None, help='sentence level or word level')

parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument("--k", type=int, default=5,
                    help="""num of samples for knn to search""")

parser.add_argument("--temperature", type=float, default=5,
                    help="""softmax temperature""")

parser.add_argument("--lambda_knn", type=float, default=5,
                    help="""weight of knn predict""")

parser.add_argument("--pretrain_path", type=str,
                    help="""Path to model files.""")

def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    test_qe_knn_token(args)

if __name__ == '__main__':
    run()
