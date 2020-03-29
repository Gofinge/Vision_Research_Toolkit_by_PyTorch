from data.transforms.transforms import Compose, RandScale, RandRotate, RandomGaussianBlur, RandomHorizontalFlip, Crop, \
    ColorJitter, Normalize, PaddingAndResize, ToTensor
from configs.defaults import TRANSFORMS


# TODO: Rebuild Transform module, make it controlled by config file!!!
@TRANSFORMS.register('voc')
class GeneralTransform(object):
    def __init__(self, cfg):
        self.mean = cfg.INPUT.PIXEL_MEAN
        self.mean = [item * 255 for item in self.mean]
        self.std = cfg.INPUT.PIXEL_STD
        self.std = [item * 255 for item in self.std]

        padding_value = [0, 0, 0]
        self.train_compose = Compose([
            RandScale([0.2, 1.0]),
            RandRotate([-10, 10], padding=padding_value, ignore_label=255),
            RandomGaussianBlur(),
            RandomHorizontalFlip(),
            Crop(cfg.INPUT.DIMS, crop_type='rand', padding=padding_value, ignore_label=255, is_remove_empty=False),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ToTensor(),
            Normalize(mean=self.mean, std=self.std)
        ])
        self.val_compose = Compose([
            PaddingAndResize(cfg.INPUT.DIMS, padding=padding_value, ignore_label=255),
            ToTensor(),
            Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, record, is_train=True):
        if is_train:
            record = self.train_compose(**record)
        else:
            record = self.val_compose(**record)
        return record


def build_transforms(cfg):
    assert cfg.DATASET.NAME.upper() in TRANSFORMS, \
        "cfg.DATASET.NAME: {} are not registered in TRANSFORMS registry".format(
            cfg.DATASET.NAME.upper
        )

    return TRANSFORMS[cfg.DATASET.NAME.upper](cfg)
