import logging
import torch
from torch.nn.parallel import DistributedDataParallel


@torch.no_grad()
def do_evaluation(cfg, model, data_loader_val, device, arguments, summary_writer):
    # get logger
    logger = logging.getLogger(cfg.NAME)
    logger.info("Start evaluation  ...")
    if isinstance(model, DistributedDataParallel):
        model = model.module
