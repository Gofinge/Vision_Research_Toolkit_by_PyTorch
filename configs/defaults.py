# Author: Xiaoyang Wu (gofinge@foxmail.com)

from yacs.config import CfgNode as CN
from utils.registry import Registry

CONFIG = Registry()
_C = CN()

# Solver
_C.SOLVER = CN()

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

# DataLoader
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKER = 4

# Model
_C.MODEL = CN()
_C.MODEL.Name = ""




@CONFIG.register('general')
def get_model_config():
    return _C.clone()
