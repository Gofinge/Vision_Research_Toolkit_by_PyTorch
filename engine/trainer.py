# Author: Xiaoyang Wu (gofinge@foxmail.com)
import logging
import time
import torch
import datetime

from utils.metric_logger import MetricLogger
from utils.miscellaneous import TqdmBar, move_to_device
from utils.comm import get_rank
from utils.summary import write_summary
from .inference import do_evaluation


def do_train(
    cfg,
    model,
    data_loader_train,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    arguments,
    summary_writer
):
    # get logger
    logger = logging.getLogger(cfg.NAME)
    logger.info("Start training ...")
    logger.info('Size of training dataset: %s' % (data_loader_train.__len__() * cfg.SOLVER.IMS_PER_BATCH))
    logger.info('Size of validation dataset: %s' % (data_loader_val.__len__() * cfg.TEST.IMS_PER_BATCH))

    model.train()

    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader_train)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    bar = TqdmBar(data_loader_train, start_iter, get_rank(), data_loader_train.__len__(),
                  description='Training', use_bar=cfg.USE_BAR)

    for iteration, record in bar.bar:
        data_time = time.time() - end
        iteration += 1
        arguments["iteration"] = iteration
        record = move_to_device(record, device)

        loss, prediction = model(record)
        optimizer.zero_grad()
        loss['total_loss'].backward()
        optimizer.step()
        scheduler.step()

        # reduce losses over all GPUs for logging purposes
        loss_reduced = {key: value.cpu().item() for key, value in loss.items()}
        meters.update(**loss_reduced)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        lr = optimizer.param_groups[0]["lr"]
        bar.set_postfix(loss)

        if iteration % cfg.SOLVER.LOGGER_PERIOD == 0 or iteration == max_iter:
            bar.clear(nolock=True)
            logger.info(
                meters.delimiter.join(
                    [
                        "iter: {iter:06d}",
                        "lr: {lr:.6f}",
                        "{meters}",
                        "eta: {eta}",
                        "mem: {memory:.0f}",
                    ]
                ).format(
                    iter=iteration,
                    lr=lr,
                    meters=str(meters),
                    eta=eta_string,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            if summary_writer:
                write_summary(summary_writer, iteration, record=loss, group='losses')
                write_summary(summary_writer, iteration, record={'lr': lr}, group='lr')

        if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if data_loader_val is not None:
                do_evaluation(cfg, model, data_loader_val, device, arguments, summary_writer)

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    bar.close()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max_iter
        )
    )










