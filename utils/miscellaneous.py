import errno
import os
import torch
from tqdm import tqdm


from .comm import is_main_process


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


def move_to_device(record, device):
    for key, value in record.items():
        if isinstance(value, torch.Tensor):
            record[key] = value.to(device)
        elif isinstance(value, (tuple, list)):
            record[key] = [v.to(device) for v in value]
        else:
            record[key] = record[key]
    return record


class TqdmBar(object):
    def __init__(self, data_loader, start_iter, distributed_rank, total=0, description='', position=0, leave=False, use_bar=True):
        if distributed_rank > 0 and use_bar:
            self.bar = tqdm(enumerate(data_loader, start_iter), total=total, position=position, leave=leave)
            self.bar.set_description(description)
        else:
            self.bar = enumerate(data_loader, start_iter)

    def set_postfix(self, info):
        if isinstance(self.bar, tqdm):
            self.bar.set_postfix(info)

    def close(self):
        if isinstance(self.bar, tqdm):
            self.bar.close()

    def clear(self, nolock=True):
        if isinstance(self.bar, tqdm):
            self.bar.clear(nolock=nolock)
