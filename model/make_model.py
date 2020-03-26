from configs.defaults import MODEL


def build_model(cfg):
    assert cfg.MODEL.NAME.upper() in MODEL, \
        "cfg.MODEL.NAME: {} are not registered in MODEL registry".format(
            cfg.MODEL.NAME.upper()
        )

    return MODEL[cfg.MODEL.upper()](cfg)
