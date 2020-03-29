import errno
import json
import logging
import os
from .comm import is_main_process
from tqdm import tqdm
from collections import OrderedDict


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


class TqdmBar(object):
    def __init__(self, data_loader=None, is_device0=None, total=0, description='', position=0, leave=False):
        if is_device0 is not None:
            self.bar = tqdm(enumerate(data_loader), total=total, position=position, leave=leave)
            self.bar.set_description(description)
        else:
            self.bar = enumerate(data_loader)

    def set_postfix(self, lr, losses):
        bar_info = OrderedDict()
        bar_info['lr'] = lr
        bar_info.update(**losses)

        if isinstance(self.bar, tqdm):
            post_str = ""
            for key in bar_info:
                post_str += "{}={}, ".format(key, self.bar.format_num(bar_info[key]))
            self.bar.set_postfix_str(post_str[0: -2])

    def close(self):
        if isinstance(self.bar, tqdm):
            self.bar.close()

    def clear(self, nolock=True):
        if isinstance(self.bar, tqdm):
            self.bar.clear(nolock=nolock)