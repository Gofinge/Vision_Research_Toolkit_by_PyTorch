# Author: Xiaoyang Wu (gofinge@foxmail.com)
import torch
import numpy as np
import cv2
import collections
import math
import random
import numbers
from configs.defaults import TRANSFORM


class Compose(object):
    # Composes data_transforms: data_transform.Compose([data_transform.RandScale([0.5, 2.0]),
    # data_transform.ToTensor()])
    def __init__(self, data_transform):
        self.data_transform = data_transform

    def __call__(self, **record):
        for t in self.data_transform:
            record = t(**record)
        return record


@TRANSFORM.register('ColorJitter')
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, center=1):
        self.brightness = [center - brightness, center + brightness]
        self.brightness[0] = max(self.brightness[0], 0)
        self.contrast = [center - contrast, center + contrast]
        self.contrast[0] = max(self.contrast[0], 0)
        self.saturation = [center - saturation, center + saturation]
        self.saturation[0] = max(self.saturation[0], 0)
        self.hue = [center - hue, center + hue]
        self.hue[0] = max(self.hue[0], 0)

    def __call__(self, **record):
        if 'image' not in record:
            raise IOError('Image not in data source.')
        image = record['image']

        brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
        image = self.adjust_brightness(image, brightness_factor)

        contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        image = self.adjust_contrast(image, contrast_factor)

        saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
        image = self.adjust_saturation(image, saturation_factor)

        hue_factor = random.uniform(self.hue[0], self.hue[1])
        image = self.adjust_hue(image, hue_factor)

        record['image'] = image
        return record

    def adjust_brightness(self, img, brightness_factor):
        im = img.astype(np.float32) * brightness_factor
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def adjust_contrast(self, img, contrast_factor):
        im = img.astype(np.float32)
        mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
        im = (1 - contrast_factor) * mean + contrast_factor * im
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def adjust_saturation(self, img, saturation_factor):
        im = img.astype(np.float32)
        degenerate = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        im = (1 - saturation_factor) * degenerate + saturation_factor * im
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def adjust_hue(self, img, hue_factor):
        im = img.astype(np.uint8)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
        hsv[..., 0] += np.uint8(hue_factor * 255)
        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return im.astype(img.dtype)


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, **record):
        if 'image' not in record:
            raise IOError('Image not in data source.')
        for key in record:
            if key == 'image':
                record[key] = torch.from_numpy(record[key].transpose(2, 0, 1))
                if not isinstance(record[key], torch.FloatTensor):
                    record[key] = record[key].float()
            elif key == 'mask':
                if record[key] is not None:
                    record[key] = torch.from_numpy(np.array(record[key])).long()
            elif key == 'bbox':
                pass
            elif key == 'keypoint':
                pass
            else:
                raise ValueError('Unknown type of data source, can not be transformed.')

        return record


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, **record):
        if 'image' not in record:
            raise IOError('Image not in data source.')
        if self.std is None:
            for t, m in zip(record['image'], self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(record['image'], self.mean, self.std):
                t.sub_(m).div_(s)
        return record


@TRANSFORM.register('Resize')
class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, a, b, mode="mm"):
        '''
        :param a: if mode == "mm", a is min_size; if mode == "wh", a is width
        :param b: if mode == "mm", b is max_size; if mode == "wh", b is height
        :param mode: "mm" or "wh"
        '''

        assert mode in ["mm", "wh"]
        self.mode = mode
        if mode == "mm":
            if not isinstance(a, (list, tuple)):
                a = (a,)
            self.min_size = a
            self.max_size = b
        else:
            self.width = a
            self.height = b

    def _get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, **record):
        if 'image' not in record:
            raise IOError('Image not in data source.')
        if self.mode == "mm":
            h, w, _ = record['image'].shape
            size = self._get_size((h, w))
        else:
            size = (self.width, self.height)

        for key in record:
            if key == 'image':
                record[key] = cv2.resize(record[key], size, interpolation=cv2.INTER_LINEAR)
            elif key == 'bbox':
                if record[key] is not None:
                    record[key] = record[key].resize(size)
            elif key == 'mask':
                if record[key] is not None:
                    record[key] = cv2.resize(record[key], size, interpolation=cv2.INTER_NEAREST)
            elif key == 'keypoint':
                if record[key] is not None and record[key].size:
                    scale_factor_x = size[1] / w
                    scale_factor_y = size[0] / h
                    record[key] = record[key] * [scale_factor_x, scale_factor_y]

            else:
                raise ValueError('Unknown type of data source, can not be transformed.')

        return record


@TRANSFORM.register('RandScale')
class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("data_transform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("data_transform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, **record):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio

        for key in record:
            if key == 'image':
                record[key] = cv2.resize(record[key], None, fx=scale_factor_x, fy=scale_factor_y,
                                         interpolation=cv2.INTER_LINEAR)
            elif key == 'bbox':
                if record[key] is not None:
                    w, h = record[key].size
                    record[key] = record[key].resize((scale_factor_x * w, scale_factor_y * h))
            elif key == 'mask':
                if record[key] is not None:
                    record[key] = cv2.resize(record[key], None, fx=scale_factor_x, fy=scale_factor_y,
                                             interpolation=cv2.INTER_NEAREST)
            elif key == 'keypoint':
                if record[key] is not None and record[key].size:
                    record[key] = record[key] * [scale_factor_x, scale_factor_y]
            else:
                raise ValueError('Unknown type of data source, can not be transformed.')
        return record


@TRANSFORM.register('PaddingAndResize')
class PaddingAndResize(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, size, padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
            self.hw_scale = 1
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
            self.hw_scale = self.crop_h * 1.0 / self.crop_w
        else:
            raise (RuntimeError("crop size error.\n"))

        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))

        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, **record):
        if 'image' not in record:
            raise IOError('Image not in data source.')
        h, w, _ = record['image'].shape
        if (h * 1.0 / w) >= (self.hw_scale):
            crop_w = round(h * 1.0 / self.hw_scale)
            crop_h = h
        else:
            crop_w = w
            crop_h = round(w * self.hw_scale)
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("data_transform.Crop() need padding while padding argument is None\n"))
            for key in record:
                if key == 'image':
                    record[key] = cv2.copyMakeBorder(record[key], pad_h_half, pad_h - pad_h_half, pad_w_half,
                                                     pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
                elif key == 'bbox':
                    if record[key] is not None:
                        record[key] = record[key].padding((pad_w_half, pad_h_half))
                elif key == 'mask':
                    if record[key] is not None:
                        record[key] = cv2.copyMakeBorder(record[key], pad_h_half, pad_h - pad_h_half, pad_w_half,
                                                         pad_w - pad_w_half, cv2.BORDER_CONSTANT,
                                                         value=self.ignore_label)
                elif key == 'keypoint':
                    if record[key] is not None and record[key].size:
                        origin_shape = record[key].shape
                        record[key] = np.reshape(record[key], (-1, 2))
                        record[key] += [pad_w_half, pad_h_half]
                        record[key] = np.reshape(record[key], origin_shape)
                else:
                    raise ValueError('Unknown type of data source, can not be transformed.')

        h, w, _ = record['image'].shape
        scale_factor_x = self.crop_w / w
        scale_factor_y = self.crop_h / h

        for key in record:
            if key == 'image':
                record[key] = cv2.resize(record[key], (self.crop_w, self.crop_h),
                                         interpolation=cv2.INTER_LINEAR)
            elif key == 'bbox':
                if record[key] is not None:
                    record[key] = record[key].resize((self.crop_w, self.crop_h))
            elif key == 'mask':
                if record[key] is not None:
                    record[key] = cv2.resize(record[key], (self.crop_w, self.crop_h),
                                             interpolation=cv2.INTER_NEAREST)
            elif key == 'keypoint':
                if record[key] is not None and record[key].size:
                    record[key] = record[key] * [scale_factor_x, scale_factor_y]
            else:
                raise ValueError('Unknown type of data source, can not be transformed.')
        return record


@TRANSFORM.register('Crop')
class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, size, crop_type='center', padding=None, ignore_label=255, is_remove_empty=True):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))

        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))

        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))

        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

        if isinstance(is_remove_empty, bool):
            self.is_remove_emoty = is_remove_empty
        else:
            raise (RuntimeError("is_remove_empty should be boolean type\n"))

    def __call__(self, **record):
        if 'image' not in record:
            raise IOError('Image not in data source.')
        h, w, _ = record['image'].shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("data_transform.Crop() need padding while padding argument is None\n"))
            for key in record:
                if key == 'image':
                    record[key] = cv2.copyMakeBorder(record[key], pad_h_half, pad_h - pad_h_half, pad_w_half,
                                                     pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
                elif key == 'bbox':
                    if record[key] is not None:
                        record[key] = record[key].padding((pad_w_half, pad_h_half))
                elif key == 'mask':
                    if record[key] is not None:
                        record[key] = cv2.copyMakeBorder(record[key], pad_h_half, pad_h - pad_h_half, pad_w_half,
                                                         pad_w - pad_w_half, cv2.BORDER_CONSTANT,
                                                         value=self.ignore_label)
                elif key == 'keypoint':
                    if record[key] is not None and record[key].size:
                        origin_shape = record[key].shape
                        record[key] = np.reshape(record[key], (-1, 2))
                        record[key] += [pad_w_half, pad_h_half]
                        record[key] = np.reshape(record[key], origin_shape)
                else:
                    raise ValueError('Unknown type of data source, can not be transformed.')

        h, w, _ = record['image'].shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)

        for key in record:
            if key == 'image':
                record[key] = record[key][h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
            elif key == 'bbox':
                if record[key] is not None:
                    record[key] = record[key].crop((w_off, h_off, w_off + self.crop_w, h_off + self.crop_h),
                                                   self.is_remove_emoty)
            elif key == 'mask':
                if record[key] is not None:
                    record[key] = record[key][h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
            elif key == 'keypoint':
                if record[key] is not None and record[key].size:
                    origin_shape = record[key].shape
                    record[key] = np.reshape(record[key], (-1, 2))
                    record[key] -= [w_off, h_off]
                    record[key] = np.reshape(record[key], origin_shape)

            else:
                raise ValueError('Unknown type of data source, can not be transformed.')

        return record


@TRANSFORM.register('RandomSampleCrop')
class RandomSampleCrop(object):
    """Crop only support record with BBox (Demo)
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self, is_remove_empty=True):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self.is_remove_empty = is_remove_empty

    def __call__(self, **record):
        from numpy import random
        if 'image' not in record:
            raise IOError('Image not in data source.')
        if 'bbox' not in record:
            raise IOError('BBox not in data source.')
        # guard against no boxes
        boxes = record['bbox'].bbox
        labels = record['bbox'].extra_fields["labels"]
        if len(boxes) == 0:
            return record
        height, width, _ = record['image'].shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return record

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float("-inf")
            if max_iou is None:
                max_iou = float("inf")

            # max trails (50)
            for _ in range(50):
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1, y1, x2, y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = self.jaccard_numpy(np.array(boxes), rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue
                # take only matching gt boxes
                current_boxes = boxes[mask]
                # take only matching gt labels
                current_labels = labels[mask]
                record["bbox"].bbox = current_boxes
                record['bbox'].extra_fields["labels"] = current_labels

                for key in record:
                    if key == 'image':
                        record[key] = record[key][rect[1]:rect[3], rect[0]:rect[2]]
                    elif key == 'bbox':
                        if record[key] is not None:
                            record[key] = record[key].crop(rect, self.is_remove_empty)
                    elif key == 'mask':
                        if record[key] is not None:
                            record[key] = record[key][rect[1]:rect[3], rect[0]:rect[2]]
                    elif key == 'keypoint':
                        if record[key] is not None and record[key].size:
                            origin_shape = record[key].shape
                            record[key] = np.reshape(record[key], (-1, 2))
                            record[key] -= rect[0: 2]
                            record[key] = np.reshape(record[key], origin_shape)
                    else:
                        raise ValueError('Unknown type of data source, can not be transformed.')
                return record

    @staticmethod
    def intersect(box_a, box_b):
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]

    def jaccard_numpy(self, box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: Multiple bounding boxes, Shape: [num_boxes,4]
            box_b: Single bounding box, Shape: [4]
        Return:
            jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
        """
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2] - box_b[0]) *
                  (box_b[3] - box_b[1]))  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]


@TRANSFORM.register('Expand')
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, **record):
        from numpy import random
        if random.randint(2):
            return record

        height, width, depth = record['image'].shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        for key in record:
            if key == 'image' or key == 'mask':
                expand_image = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=record[key].dtype)
                expand_image[int(top):int(top + height), int(left):int(left + width)] = record[key]
                record[key] = expand_image
            elif key == 'bbox':
                if record[key] is not None:
                    record[key].bbox[:, :2] += torch.tensor([int(left), int(top)], dtype=torch.float)
                    record[key].bbox[:, 2:] += torch.tensor([int(left), int(top)], dtype=torch.float)
                    record[key].image_size = (height * ratio, width * ratio)
            elif key == 'keypoint':
                if record[key] is not None and record[key].size:
                    pass
            else:
                raise ValueError('Unknown type of data source, can not be transformed.')
        return record


@TRANSFORM.register('RandRotate')
class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("data_transform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, **record):
        if random.random() > self.p:
            return record

        angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
        if 'image' not in record:
            raise IOError('Image not in data source.')
        h, w, _ = record['image'].shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        for key in record:
            if key == 'image':
                record[key] = cv2.warpAffine(record[key], matrix, (w, h), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=self.padding)
            elif key == 'mask':
                if record[key] is not None:
                    record[key] = cv2.warpAffine(record[key], matrix, (w, h), flags=cv2.INTER_NEAREST,
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=self.ignore_label)
            elif key == 'bbox':
                if record[key] is not None:
                    record[key] = record[key].rotate(angle, matrix)
            elif key == 'keypoint':
                if record[key] is not None and record[key].size:
                    origin_shape = record[key].shape
                    record[key] = np.reshape(record[key], (-1, 2))
                    for point in record[key]:
                        point[0], point[1] = self.point_rotate(point[0], point[1], matrix)
                    record[key] = np.reshape(record[key], origin_shape)

            else:
                raise ValueError('Unknown type of data source, can not be transformed.')

        return record

    def point_rotate(self, x, y, matrix):
        new_x = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
        new_y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
        return new_x, new_y


@TRANSFORM.register('RandomHorizontalFlip')
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, **record):
        if random.random() > self.p:
            return record

        if 'image' not in record:
            raise IOError('Image not in data source.')
        _, w, _ = record['image'].shape
        for key in record:
            if key == 'image':
                record[key] = cv2.flip(record[key], 1)
            elif key == 'bbox':
                if record[key] is not None:
                    record[key] = record[key].transpose(0)
            elif key == 'mask':
                if record[key] is not None:
                    record[key] = cv2.flip(record[key], 1)
            elif key == 'keypoint':
                if record[key] is not None and record[key].size:
                    origin_shape = record[key].shape
                    record[key] = np.reshape(record[key], (-1, 2))
                    for point in record[key]:
                        point[0] = w - point[0]
                    record[key] = np.reshape(record[key], origin_shape)

            else:
                raise ValueError('Unknown type of data source, can not be transformed.')
        return record


@TRANSFORM.register('RandomVerticalFlip')
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, **record):
        if random.random() > self.p:
            return record

        if 'image' not in record:
            raise IOError('Image not in data source.')
        h, _, _ = record['image'].shape
        for key in record:
            if key == 'image':
                record[key] = cv2.flip(record[key], 0)
            elif key == 'bbox':
                if record[key] is not None:
                    record[key] = record[key].transpose(1)
            elif key == 'mask':
                if record[key] is not None:
                    record[key] = cv2.flip(record[key], 0)
            elif key == 'keypoint':
                if record[key] is not None and record[key].size:
                    origin_shape = record[key].shape
                    record[key] = np.reshape(record[key], (-1, 2))
                    for point in record[key]:
                        point[1] = h - point[1]
                    record[key] = np.reshape(record[key], origin_shape)

            else:
                raise ValueError('Unknown type of data source, can not be transformed.')
        return record


@TRANSFORM.register('RandomGaussianBlur')
class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, **record):
        if random.random() < 0.5:
            if 'image' not in record:
                raise IOError('Image not in data source.')
            record['image'] = cv2.GaussianBlur(record['image'], (self.radius, self.radius), 0)
        return record


class Transforms(object):
    def __init__(self, trans_cfg):
        transforms = []
        for transform in trans_cfg:
            name = transform['name']
            args = transform['arg']
            assert name in TRANSFORM, \
                "Transform: {} is not registered in OPTIMIZER registry".format(
                    name
                )
            transforms.append(TRANSFORM[name](**args))
        self.compose = Compose(transforms)

    def __call__(self, record):
        record = self.compose(**record)
        return record
