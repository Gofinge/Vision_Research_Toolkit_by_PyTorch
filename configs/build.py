# Author: Xiaoyang Wu (gofinge@foxmail.com)
import yacs.config as config
from .defaults import CONFIG


def make_config(config_file, config_list):
    with open(config_file, 'r') as f:
        cfg_from_file = config.load_cfg(f)

    assert cfg_from_file.MODEL.NAME.upper() in CONFIG, \
        "Defaults config for {} model are not refistered in registry".format(
            cfg_from_file.MODEL.NAME.upper()
        )

    cfg = CONFIG[cfg_from_file.MODEL.NAME.upper()]()
    cfg.merge_from_other_cfg(cfg_from_file)
    cfg.merge_from_list(config_file)
    return cfg.clone()