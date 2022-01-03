import argparse

from src.main import knn_analysis

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--feature_prefix', type=str, default="/home/yanym/save/enzh/faiss_index/",
                    help="The path for feature_prefix")

parser.add_argument('--knn_log_prefix', type=str, default="/home/yanym/save/enzh/knn_log/",
                    help="The path for knn_log_prefix")

parser.add_argument('--search_level', type=str, default="sent",
                    help="sent or token level for knn to search for")

parser.add_argument('--feature_model', type=str, default="directqe",
                    help="use which model to extract feature")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument("--test_id", type=int, default=0,
                    help="""id of focused test sample""")

parser.add_argument("--test_token", type=int, default=0,
                    help="""id of focused token of focused test sample""")

parser.add_argument("--pretrain_path", type=str,
                    help="""Path to model files.""")



def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    knn_analysis(args)

if __name__ == '__main__':
    run()
