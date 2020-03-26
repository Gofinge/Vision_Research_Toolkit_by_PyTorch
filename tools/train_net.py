# Author: Xiaoyang Wu (gofinge@foxmail.com)
r"""
Basic training script for PyTorch
"""

import os
import argparse
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel, DataParallel
from time import localtime, strftime

from utils.comm import synchronize, get_rank
from utils.miscellaneous import mkdir, save_config
from utils.logger import setup_logger
from utils.collect_env import collect_env_info
from configs.build import make_config
from model.make_model import build_model
from solver.build import make_optimizer, make_lr_scheduler


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
        cfg.SAVER.NAME = cfg.LOADER.NAME
    else:
        cfg.LOADER.DIR = ''
        cfg.LOADER.NAME = ''

    if len(cfg.SAVER.NAME) is 0:
        cfg.SAVER.NAME = strftime("%Y-%m-%d-%H-%M-%S", localtime())

    cfg.freeze()

    output_dir = os.path.join(cfg.SAVER.DIR, cfg.SAVER.NAME)
    mkdir(output_dir)

    # Init logger
    logger = setup_logger(cfg.NAME, output_dir, get_rank())
    logger.info("Using {} GPU".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info ...")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(output_dir, os.path.basename(args.config_file))
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    train(cfg, args.local_rank, args.distributed)
    return


def train(cfg, local_rank, distributed):
    # build model
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # build solver
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = os.path.join(cfg.SAVER.DIR, cfg.SAVER.NAME)

    save_to_disk = get_rank() == 0
    checkpointer =

    # TODO: Should we separate Loader.dir and Saver.dir?
    # load_dir = os.path.join(cfg.LOADER.DIR, cfg.LOADER.NAME)
    # save_dir = os.path.join(cfg.SAVER.DIR, cfg.LOADER.NAME)
    # saver = Saver(model, save_dir, load_dir, local_rank)


if __name__ == "__main__":
    main()
