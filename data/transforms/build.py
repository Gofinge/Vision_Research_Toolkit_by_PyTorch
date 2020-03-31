from .transforms import Transforms


def build_transforms(cfg, is_train):
    if is_train:
        trans_cfg = cfg.TRANSFORM.TRAIN
    else:
        trans_cfg = cfg.TRANSFORM.VAL
    return Transforms(trans_cfg)
