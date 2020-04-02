import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
import cv2
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from utils.image_list import ImageList
from utils.registry import DATASET
import warnings


def get_class_name_and_color(idx, dataset_name):
    try:
        name = DATASET[dataset_name].CLASSES_NAMES[idx]
    except AttributeError:
        warnings.warn("No Attribute CLASSES_NAMES in {}".format(dataset_name))
        name = ""

    try:
        color = DATASET[dataset_name].CLASSES_COLORS[idx]
    except AttributeError:
        warnings.warn("No Attribute CLASSES_COLORS in {}".format(dataset_name))
        general_classes_colors = (torch.tensor(range(200))[:, None] *
                                  torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1]) % 255).numpy().astype("uint8")
        color = general_classes_colors[idx]

    return name, color


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be unnormalized.
        Returns:
            Tensor: UnNormalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def project(pt3d, K, RT):
    P = np.dot(K, RT)
    n = pt3d.shape[0]
    pt3d = np.hstack((pt3d, np.ones((n, 1))))
    pt2d = np.dot(P, np.transpose(pt3d))
    pt2d[0, :] = pt2d[0, :] / pt2d[2, :]
    pt2d[1, :] = pt2d[1, :] / pt2d[2, :]
    return pt2d[0:2, :]


def vertice_sampling(vertices, sample_ratio=0.1):
    n = vertices.shape[0]
    selected = np.random.choice(n, size=int(n * sample_ratio))
    return vertices[selected, :]


def vis_pose(image, class_ids, models, instrinsic_matrix, poses):
    for c in class_ids:
        vertices_2d = project(models[c], instrinsic_matrix, poses)
        image_pose = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image_pose)
        draw.point(np.transpose(vertices_2d).flatten().tolist(), fill=(255, 255, 0, 255))
        image = Image.alpha_composite(image, image_pose)


def visualize_mask(cfg, record, prediction, iteration, logger=None):
    unloader = transforms.ToPILImage()
    mean = cfg.INPUT.PIXEL_MEAN
    std = cfg.INPUT.PIXEL_STD
    unorm = UnNormalize(mean, std)

    image = record['image']
    image = torch.squeeze(image)
    image = unorm(image)
    image = image.numpy().astype(np.uint8)
    image = image.transpose(1, 2, 0)
    image = Image.fromarray(image)

    pred = np.squeeze(np.array(prediction))
    pred = decode_mask(pred, cfg.DATASET.NAME, cfg.DATASET.NUM_CLASS)
    pred = unloader(np.array(pred, dtype=np.uint8))

    target_image = Image.blend(image, pred, 0.4)

    file_name = '%09d-pred.png' % iteration
    file_name = os.path.join(cfg.SAVER.SAVER_DIR, file_name)
    target_image.save(file_name)

    if logger is not None:
        logger.info('Writing image to %s', file_name)


def visualize_bbox_from_BoxLists(record):
    import cv2
    import numpy as np
    image = record['image']
    label = record['bbox']
    image = image.numpy().transpose(1, 2, 0)
    image = np.asarray(image, dtype=np.uint8)
    temp = image.copy()
    if 'labels' in label.extra_fields:
        boxes = label.bbox.numpy()
        class_id = label.extra_fields['labels'].numpy()
        for box, cls in zip(boxes, class_id):
            color = (0, 0, 255) if cls == [1] else (0, 255, 0)
            cv2.rectangle(temp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.imshow('image', temp)
        cv2.waitKey(0)


def draw_bounding_box(cfg, image, dets, input_normalized=True, threshold=0.3):
    image = torch.squeeze(image)
    if input_normalized:
        unorm = UnNormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        image = unorm(image)
    unloader = torchvision.transforms.ToPILImage()
    image_pil = unloader(image)
    dets_numpy = np.squeeze(dets.detach().numpy())
    nbox = dets_numpy.shape[0]
    drawer = ImageDraw.Draw(image_pil)

    for i in range(nbox):
        x1, y1, x2, y2 = dets_numpy[i, 0:4]
        if dets_numpy[i, 4] > threshold:
            label_names, label_colours = get_class_name_and_color(int(dets_numpy[i, 5]), cfg.DATASET.NAME)
            drawer.rectangle((x1, y1, x2, y2), outline=tuple(label_colours), width=2)
            drawer.text((x1, y1 - 13), label_names, fill=tuple(label_colours))
    return image_pil


def visualize_bbox_from_BoxNumpy(cfg, record, prediction, index, logger=None):
    image = record['image']
    if isinstance(image, ImageList):
        image = image.tensors
    image_pil = draw_bounding_box(cfg, image, prediction, threshold=0.3)

    file_name = '%09d-pred.png' % index
    file_name = os.path.join(cfg.SAVER.SAVER_DIR, file_name)
    image_pil.save(file_name)

    if logger is not None:
        logger.info('Writing image to %s', file_name)


def visualize_bbox_from_heatmap(image, heatmap, wh, reg, ind):
    image = image.cpu().numpy().transpose(0, 2, 3, 1)
    image = np.array(image, dtype=np.uint8)
    image = image.copy()
    nbatch, height, width, _ = image.shape
    heatmap = heatmap.cpu().numpy().transpose(0, 2, 3, 1)
    ind = ind.cpu().numpy()
    wh = wh.cpu().numpy()
    reg = reg.cpu().numpy()
    for count in range(nbatch):
        temp = image[count]
        for ind_i, wh_i, reg_i in zip(ind[count], wh[count], reg[count]):
            y = (int(ind_i / (width / 4)) + reg_i[1]) * 4
            x = (int(ind_i % (width / 4)) + reg_i[0]) * 4
            w, h = wh_i * 4
            color = (0, 255, 255)
            left = int(x - w / 2)
            top = int(y - h / 2)
            right = int(x + w / 2)
            bottom = int(y + h / 2)
            cv2.rectangle(temp, (left, top), (right, bottom), color, 2)
        cv2.imshow('heatmap', heatmap[count])
        cv2.imshow('image', temp)
        cv2.waitKey(0)


def decode_mask(label_mask, dataset_name='ade20k', num_class=151, plot=False):
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for cls in range(num_class):
        label_names, label_colours = get_class_name_and_color(cls, dataset_name)
        r[label_mask == cls] = label_colours[0]
        g[label_mask == cls] = label_colours[1]
        b[label_mask == cls] = label_colours[2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
