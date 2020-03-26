# Author: Xiaoyang Wu (gofinge@foxmail.com)

from yacs.config import CfgNode as CN
from utils.registry import Registry

_C = CN()
_C.NAME = "VRT by PyTorch"      # Which will be the name of logger

# Solver
_C.SOLVER = CN()
_C.SOLVER.NAME = "SGD"  # now support "SGD", "Adam"

# Saver
_C.SAVER = CN()
_C.SAVER.NAME = ""      # Name of saver in SAVER.NAME, use time as name if name is ""(default)
_C.SAVER.DIR = "log"        # Dir of saver, support absolute path(Linux and MacOS) and relative path
_C.SAVER.SAVE_EVERY_EPOCH = 5

# Loader
_C.LOADER = CN()
_C.LOADER.NAME = ""
_C.LOADER.DIR = "log"       # Dir of saver, support absolute path(Linux and MacOS) and relative path
_C.LOADER.LOAD_EPOCH = 0        # If LOAD_EPOCH is 0, the last checkpoint will be loaded
_C.LOADER.CONTINUE = False      # If continue is true, the model will start training from loaded checkpoint

# Dataset
_C.DATASET = CN()
_C.DATASET.NAME = ""
_C.DATASET.DATA_TYPE = []

# DataLoader
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKER = 4

# Model
_C.MODEL = CN()
_C.MODEL.NAME = ""


MODEL_CONFIG = Registry()
OPTIMIZER_CONFIG = Registry()

MODEL = Registry()
OPTIMIZER = Registry()


@OPTIMIZER_CONFIG.register("SGD")
def get_optimizer_config():
    _C.SOLVER.BASE_LR = 0.001
    _C.SOLVER.BIAS_LR_FACTOR = 2
    _C.SOLVER.MOMENTUM = 0.9
    _C.SOLVER.WEIGHT_DECAY = 0.0005
    _C.SOLVER.WEIGHT_DECAY_BIAS = 0


@OPTIMIZER_CONFIG.register("ADAM")
def get_optimizer_config():
    _C.SOLVER.BASE_LR = 0.001
    _C.SOLVER.BETAS = (0.9, 0.999)
    _C.SOLVER.EPS = 1e-08
    _C.SOLVER.WEIGHT_DECAY = 0.0005


@MODEL_CONFIG.register("PSP")
def get_model_config():
    _C.DATASET.DATA_TYPE = ["MASK"]


def get_default_config():
    return _C.clone()


