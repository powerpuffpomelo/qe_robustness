import argparse

from src.main import train_probe
from . import auto_mkdir

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--log_path', type=str, default="./log",
                    help="The path for saving tensorboard logs. Default is ./log")

parser.add_argument('--saveto', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--debug', action="store_true",
                    help="Use debug mode.")
                    
parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--pretrain_path', type=str, default=None, help="The path for pretrained model.")

parser.add_argument("--predefined_config", type=str, default=None,
                    help="""Use predefined configuration.""")

parser.add_argument('--representation', type=str, default="hidden",
                    help="representation type as probe input")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(args.log_path)
    auto_mkdir(args.saveto)

    train_probe(args)

if __name__ == '__main__':
    run()
