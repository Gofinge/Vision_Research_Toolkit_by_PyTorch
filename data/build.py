# Author: Xiaoyang Wu (gofinge@foxmail.com)
import bisect
import copy
import logging

import torch.utils.data
from utils.comm import get_world_size
from utils.imports import import_file
from data.transforms.build import build_transforms


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0, is_for_period=False):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    paths_catalog = import_file(
        "configs.paths_catalog", cfg.PATHS_CATALOG, True
    )
    dataset_catalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST



