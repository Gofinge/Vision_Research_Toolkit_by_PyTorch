# Author: Xiaoyang Wu (gofinge@foxmail.com)
from configs.defaults import MODEL


# register SSD
@MODEL.register('ssd')
def generate_model(cfg):
    from model.det.ssd.ssd import SSDDetector as SSDNet
    from model.det.ssd.ssd import NetWrapper as SSDNetWrapper
    return SSDNetWrapper(SSDNet(cfg))


def build_model(cfg):
    assert cfg.MODEL.NAME.upper() in MODEL, \
        "cfg.MODEL.NAME: {} are not registered in MODEL registry".format(
            cfg.MODEL.NAME.upper()
        )

    return MODEL[cfg.MODEL.upper()](cfg)
