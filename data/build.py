# Author: Xiaoyang Wu (gofinge@foxmail.com)
import bisect
import copy
import logging

import torch.utils.data
from utils.comm import get_world_size


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0, is_for_period=False):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH