# Author: Xiaoyang Wu (gofinge@foxmail.com)
r"""
Basic training script for PyTorch
"""

import os
import argparse

import torch
import torch.distributed
from utils.communication import synchronize, get_rank
from utils.common import mkdir
from configs.build import make_config
from time import localtime, strftime


def main():
    # Add augments
    parser = argparse.ArgumentParser(description="Vision Research Toolkit by PyTorch")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    # make config
    cfg = make_config(args.config_file, args.opts)

    # obtain absolute dir of project
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    if len(cfg.SAVER.DIR) != 0:
        if cfg.SAVER.DIR[0] is not os.sep:
            # if the saver_dir is not absolute dir
            cfg.SAVER.DIR = os.path.join(project_dir, cfg.SAVER.DIR)
    else:
        cfg.SAVER.DIR = os.path.join(project_dir, 'log')

    if len(cfg.LOADER.DIR) != 0:
        if cfg.LOADER.DIR[0] is not os.sep:
            # if the loader_dir is not absolute dir
            cfg.LOADER.DIR = os.path.join(project_dir, cfg.LOADER.DIR)

    if cfg.LOADER.CONTINUE:
        cfg.SAVER.DIR = cfg.LOADER.DIR
    else:
        # only run the following instruction on GPU 0
        if get_rank() == 0:
            if len(cfg.SAVER.NAME) is 0:
                cfg.SAVER.NAME = strftime("%Y-%m-%d-%H-%M-%S", localtime())
            mkdir(os.path.join(cfg.SAVER.DIR, cfg.SAVER.NAME))
        cfg.LOADER.DIR = ''

    cfg.freeze()

