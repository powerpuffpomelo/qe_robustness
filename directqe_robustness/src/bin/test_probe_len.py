import argparse

from src.main import test_probe_len
from . import auto_mkdir

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--log_path', type=str, default="./log",
                    help="The path for saving tensorboard logs. Default is ./log")

parser.add_argument('--saveto', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--pretrain_path', type=str, default=None, 
                    help="The path for pretrained model.")

parser.add_argument('--pretrain_path_probe', type=str, default=None, 
                    help="The path for pretrained model.")

parser.add_argument("--predefined_config", type=str, default=None,
                    help="""Use predefined configuration.""")

parser.add_argument('--representation', type=str, default="hidden",
                    help="representation type as probe input")

parser.add_argument('--len_to_pred', type=str, default="pe",
                    help="predict length of which sequence")
                    

def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(args.log_path)
    auto_mkdir(args.saveto)

    test_probe_len(args)

if __name__ == '__main__':
    run()
