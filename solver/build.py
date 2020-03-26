import torch
from configs.defaults import OPTIMIZER
from .lr_scheduler import WarmupMultiStepLR


@OPTIMIZER.register("SGD")
def _make_sgd(cfg, model):
    params = []
    for key, value in model.nameed_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


@OPTIMIZER.register("ADAM")
def _make_adam(cfg, model):
    optimizer = torch.optim.Adam(model.parameters(), cfg.SOLVER.BASE_LR,
                                 betas=cfg.SOLVER.BETAS, eps=cfg.SOLVER.EPS, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return optimizer


def make_optimizer(cfg, model):
    assert cfg.SOLVER.NAME.upper() in OPTIMIZER, \
        "cfg.SOLVER.NAME: {} are not registered in OPTIMIZER registry".format(
            cfg.SOLVER.NAME.upper
        )

    return OPTIMIZER[cfg.SOLVER.NAME.upper](cfg, model)


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
