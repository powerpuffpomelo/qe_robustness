import argparse

from src.main import finetune_qe_robust_aug_dynamic
from . import auto_mkdir

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,
                    help="The name of the model. Will alse be the prefix of saving archives.")

parser.add_argument('--qe_model_name', type=str,
                    help="The name of the model. Will alse be the prefix of saving archives.")

parser.add_argument('--reload', action="store_true",
                    help="Whether to restore from the latest archives.")

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--log_path', type=str, default="./log",
                    help="The path for saving tensorboard logs. Default is ./log")

parser.add_argument('--seed', type=int, default="1234")

parser.add_argument('--saveto', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--debug', action="store_true",
                    help="Use debug mode.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--pretrain_generator_mlm_path', type=str, default=None, help="The path for pretrained model.")

parser.add_argument('--pretrain_path', type=str, default=None, help="The path for pretrained model.")

parser.add_argument('--level', type=str, default=None, help='sentence level or word level')

parser.add_argument("--multi_gpu", action="store_true",
                    help="""Running on multiple GPUs (No need to manually add this option).""")

parser.add_argument("--shared_dir", type=str, default="/tmp",
                    help="""Shared directory across nodes. Default is '/tmp'""")

parser.add_argument("--predefined_config", type=str, default=None,
                    help="""Use predefined configuration.""")

parser.add_argument("--align_bpe_path_mt2src", type=str, default=None,
                    help="""align_bpe_path""")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(args.log_path)
    auto_mkdir(args.saveto)

    finetune_qe_robust_aug_dynamic(args)

if __name__ == '__main__':
    run()
