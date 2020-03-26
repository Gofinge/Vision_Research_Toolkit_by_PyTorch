# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2

from torch.utils.collect_env import get_pretty_env_info


def get_cv2_version():
    return "\n        OpenCV ({})".format(cv2.__version__)


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_cv2_version()
    return env_str
