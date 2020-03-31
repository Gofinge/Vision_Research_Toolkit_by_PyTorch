# Author: Xiaoyang Wu (gofinge@foxmail.com)

import os
from yacs.config import CfgNode as CN
from utils.registry import Registry

# Miscellaneous
_C = CN()
_C.NAME = "VRT by PyTorch"      # Which will be the name of logger
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
_C.USE_BAR = True       # Whether use tqdm bar
_C.SUMMARY_WRITER = "tensorboard"

# Solver
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 8
_C.SOLVER.MAX_ITER = 100000
_C.SOLVER.CHECKPOINT_PERIOD = 2500
_C.SOLVER.LOGGER_PERIOD = 20
_C.SOLVER.EVALUATE = True    # whether do evaluation while saving checkpoints
_C.SOLVER.OPTIMIZER = CN()
_C.SOLVER.OPTIMIZER.NAME = "SGD"  # now support "SGD", "Adam"
_C.SOLVER.SCHEDULER = CN()
_C.SOLVER.SCHEDULER.NAME = "WarmupMultiStepLR"  # now only support "WarmupMultiStepLR"

# Test
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8

# CheckPointer
_C.CHECKPOINTER = CN()
_C.CHECKPOINTER.NAME = ""      # Name of saver in SAVER.NAME, use time as name if name is ""(default)
_C.CHECKPOINTER.DIR = "log"        # Dir of saver, support absolute path(Linux and MacOS) and relative path
_C.CHECKPOINTER.SAVE_EPOCH = 5
_C.CHECKPOINTER.LOAD_NAME = ""  # if load_name is "", the checkpointer will load the latest checkpoint

# Dataset
_C.DATASET = CN()
_C.DATASET.TRAIN = ()      # TrainSet name in path_catalog.py
_C.DATASET.TEST = ()       # TestSet name in path_catalog.py
_C.DATASET.DATA_TYPE = []   # ["mask", "bbox", "keypoint"]
_C.DATASET.NUM_CLASS = 1

# Transform
# Transform can be set in yaml config file, follow the instruction in template
_C.TRANSFORM = CN(new_allowed=True)

# DataLoader
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKER = 4
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = False

# Model
_C.MODEL = CN()
_C.MODEL.NAME = ""


MODEL_CONFIG = Registry()
OPTIMIZER_CONFIG = Registry()
SCHEDULER_CONFIG = Registry()

MODEL = Registry()
OPTIMIZER = Registry()
SCHEDULER = Registry()
TRANSFORM = Registry()


@OPTIMIZER_CONFIG.register("SGD")
def get_optimizer_config():
    _C.SOLVER.OPTIMIZER.BASE_LR = 0.001
    _C.SOLVER.OPTIMIZER.BIAS_LR_FACTOR = 2
    _C.SOLVER.OPTIMIZER.MOMENTUM = 0.9
    _C.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.0005
    _C.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS = 0


@OPTIMIZER_CONFIG.register("ADAM")
def get_optimizer_config():
    _C.SOLVER.OPTIMIZER.BASE_LR = 0.001
    _C.SOLVER.OPTIMIZER.BETAS = (0.9, 0.999)
    _C.SOLVER.OPTIMIZER.EPS = 1e-08
    _C.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.0005


@SCHEDULER_CONFIG.register("WARMUPMULTISTEPLR")
def get_scheduler_config():
    _C.SOLVER.SCHEDULER.STEPS = (30000, )
    _C.SOLVER.SCHEDULER.GAMMA = 0.1
    _C.SOLVER.SCHEDULER.WARMUP_FACTOR = 1.0 / 3
    _C.SOLVER.SCHEDULER.WARMUP_ITERS = 500
    _C.SOLVER.SCHEDULER.WARMUP_METHOD = "linear"


@MODEL_CONFIG.register("PSP")
def get_model_config():
    _C.DATASET.DATA_TYPE = ["MASK"]


def get_default_config():
    return _C.clone()


