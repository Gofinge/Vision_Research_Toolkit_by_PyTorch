# Author: Xiaoyang Wu (gofinge@foxmail.com)
import yacs.config as config
from .defaults import MODEL_CONFIG, OPTIMIZER_CONFIG, get_default_config


def make_config(config_file, config_list):
    with open(config_file, 'r') as f:
        cfg_from_file = config.load_cfg(f)

    assert cfg_from_file.SOLVER.NAME.upper() in OPTIMIZER_CONFIG, \
        "Defaults config for {} optimizer are not registered in OPTIMIZER_CONFIG registry".format(
            cfg_from_file.SOLVER.NAME.upper()
        )

    assert cfg_from_file.MODEL.NAME.upper() in MODEL_CONFIG, \
        "Defaults config for {} model are not registered in MODEL_CONFIG registry".format(
            cfg_from_file.MODEL.NAME.upper()
        )

    OPTIMIZER_CONFIG[cfg_from_file.SOLVER.NAME.upper()]()
    MODEL_CONFIG[cfg_from_file.MODEL.NAME.upper()]()
    cfg = get_default_config()

    cfg.merge_from_other_cfg(cfg_from_file)
    cfg.merge_from_list(config_list)
    return cfg.clone()
