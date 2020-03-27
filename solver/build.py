import torch
from configs.defaults import OPTIMIZER, SCHEDULER
from .lr_scheduler import WarmupMultiStepLR


# Make and register Optimizer


@OPTIMIZER.register("SGD")
def _make_sgd(cfg, model):
    params = []
    for key, value in model.nameed_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR
        weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM)
    return optimizer


@OPTIMIZER.register("ADAM")
def _make_adam(cfg, model):
    optimizer = torch.optim.Adam(model.parameters(), cfg.SOLVER.OPTIMIZER.BASE_LR,
                                 betas=cfg.SOLVER.OPTIMIZER.BETAS, eps=cfg.SOLVER.OPTIMIZER.EPS, weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY)
    return optimizer


def make_optimizer(cfg, model):
    assert cfg.SOLVER.OPTIMIZER.NAME.upper() in OPTIMIZER, \
        "cfg.SOLVER.OPTIMIZER.NAME: {} are not registered in OPTIMIZER registry".format(
            cfg.SOLVER.OPTIMIZER.NAME.upper
        )

    return OPTIMIZER[cfg.SOLVER.OPTIMIZER.NAME.upper](cfg, model)


# Make and register Scheduler


@SCHEDULER.register("WARMUPMUTISTEPLR")
def make_warmup_muti_step_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.SCHEDULER.STEPS,
        cfg.SOLVER.SCHEDULER.GAMMA,
        warmup_factor=cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.SCHEDULER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.SCHEDULER.WARMUP_METHOD,
    )


def make_lr_scheduler(cfg, optimizer):
    assert cfg.SOLVER.SCHEDULER.NAME.upper() in OPTIMIZER, \
        "cfg.SOLVER.SCHEDULER.NAME: {} are not registered in SCHEDULER registry".format(
            cfg.SOLVER.SCHEDULER.NAME.upper
        )

    return SCHEDULER[cfg.SOLVER.SCHEDULER.NAME.upper](cfg, optimizer)
