# Author: Xiaoyang Wu (gofinge@foxmail.com)
r"""
Basic training script for PyTorch
"""

import os
import argparse
import logging
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from time import localtime, strftime

from utils.comm import synchronize, get_rank
from utils.miscellaneous import mkdir, save_config
from utils.logger import setup_logger
from utils.collect_env import collect_env_info
from utils.checkpoint import Checkpointer
from utils.summary import make_summary_writer
from configs.build import make_config
from model.make_model import build_model
from solver.build import make_optimizer, make_lr_scheduler
from data.build import make_data_loader
from engine.trainer import do_train


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

    if cfg.CHECKPOINTER.DIR:
        if cfg.CHECKPOINTER.DIR[0] is not os.sep:
            # if the saver_dir is not absolute dir
            cfg.CHECKPOINTER.DIR = os.path.join(project_dir, cfg.CHECKPOINTER.DIR)
    else:
        cfg.CHECKPOINTER.DIR = os.path.join(project_dir, 'log')

    if not cfg.CHECKPOINTER.NAME:
        cfg.CHECKPOINTER.NAME = strftime("%Y-%m-%d-%H-%M-%S", localtime())

    cfg.freeze()

    save_dir = os.path.join(cfg.CHECKPOINTER.DIR, cfg.CHECKPOINTER.NAME)
    mkdir(save_dir)

    # Init logger
    logger = setup_logger(cfg.NAME, save_dir, get_rank())
    logger.info("Using {} GPU".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info ...")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(save_dir, os.path.basename(args.config_file))
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    train(cfg, args.local_rank, args.distributed)
    return


def train(cfg, local_rank, distributed):
    logger = logging.getLogger(cfg.NAME)
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

    arguments = {"iteration": 0}

    save_dir = os.path.join(cfg.CHECKPOINTER.DIR, cfg.CHECKPOINTER.NAME)

    save_to_disk = get_rank() == 0
    checkpointer = Checkpointer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        save_dir=save_dir, save_to_disk=save_to_disk, logger=logger
    )
    extra_checkpoint_data = checkpointer.load(cfg.CHECKPOINTER.LOAD_NAME)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    evaluate = cfg.SOLVER.EVALUATE
    if evaluate:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    save_to_disk = get_rank() == 0
    if cfg.SUMMARY_WRITER and save_to_disk:
        save_dir = os.path.join(cfg.CHECKPOINTER.DIR, cfg.CHECKPOINTER.NAME)
        summary_writer = make_summary_writer(cfg.SUMMARY_WRITER, save_dir, model_name=cfg.MODEL.NAME)
    else:
        summary_writer = None

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        arguments,
        summary_writer
    )

    return model


if __name__ == "__main__":
    main()
