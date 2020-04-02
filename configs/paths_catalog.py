# Author: Xiaoyang Wu (gofinge@foxmail.com)
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/home/gofinge/Documents/data"
    DATASETS = {
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "voc/VOC2007",
            "split": "trainval"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "voc/VOC2012",
            "split": "trainval"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
        }
    }
    FACTORY = {"voc": "PascalVOCDataset"}

    @staticmethod
    def get(name):
        if "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            args = DatasetCatalog.DATASETS[name]
            args["data_dir"] = os.path.join(data_dir, args["data_dir"])
            return dict(
                name='voc',     # file name in data/datasets
                args=args,
            )
