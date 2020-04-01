# Author: Xiaoyang Wu (gofinge@foxmail.com)
import os


def make_summary_writer(name, save_dir, model_name=None):
    if name.lower() == "pavi":
        from pavi import SummaryWriter
        writer = SummaryWriter(os.path.join(save_dir, "pavi"), model=model_name)
        return writer

    elif name.lower() == "tensorboard":
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join(save_dir, "tensorboard"))
        return writer
    else:
        raise ValueError('Unknown visualization tool! Please confirm your config file.')


def write_summary(summary_writer, iteration, record=None, group=None):
    if isinstance(record, dict):
        for key, value in record.items():
            if group is not None:
                key = group + '/' + key
            summary_writer.add_scalar(key, value, iteration)