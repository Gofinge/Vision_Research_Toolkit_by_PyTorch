from yacs.config import CfgNode as CN


def ssd_config(cfg):
    cfg.MODEL.THRESHOLD = 0.5
    cfg.MODEL.NEG_POS_RATIO = 3
    cfg.MODEL.CENTER_VARIANCE = 0.1
    cfg.MODEL.SIZE_VARIANCE = 0.2

    # ---------------------------------------------------------------------------- #
    # Backbone
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = 'vgg'
    cfg.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
    cfg.MODEL.BACKBONE.PRETRAINED = True

    # -----------------------------------------------------------------------------
    # PRIORS
    # -----------------------------------------------------------------------------
    cfg.MODEL.PRIORS = CN()
    cfg.MODEL.PRIORS.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
    cfg.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 100, 300]
    cfg.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
    cfg.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
    cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    # When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
    # #boxes = 2 + #ratio * 2
    cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
    cfg.MODEL.PRIORS.CLIP = True

    # -----------------------------------------------------------------------------
    # Box Head
    # -----------------------------------------------------------------------------
    cfg.MODEL.BOX_HEAD = CN()
    cfg.MODEL.BOX_HEAD.NAME = 'SSDBoxHead'
    cfg.MODEL.BOX_HEAD.PREDICTOR = 'SSDBoxPredictor'
    return cfg
