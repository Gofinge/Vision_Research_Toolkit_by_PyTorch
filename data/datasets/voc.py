# Author: Xiaoyang Wu (gofinge@foxmail.com)
import os
import torch
import torch.utils.data
import cv2
import numpy as np
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from data.structures.bounding_box import BoxList


class PascalVOCDataset(torch.utils.data.Dataset):
    CLASSES_NAMES = (
        "__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    )

    CLASSES_COLORS = (
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
        [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
    )

    def __init__(self, data_dir, split, data_type=None, use_difficult=False, transforms=None):
        self.data_type = data_type
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._anno_path = os.path.join(self.root, "Annotations", "%s.xml")
        self._img_path = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        self._mask_path = os.path.join(self.root, "SegmentationClass", "%s.png")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PascalVOCDataset.CLASSES_NAMES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        image = cv2.imread(self._img_path % img_id, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        record = {'image': image}

        img_id = self.ids[index]
        if 'bbox' in self.data_type:
            anno = ET.parse(self._anno_path % img_id).getroot()
            anno = self._preprocess_annotation(anno)

            height, width = anno["im_info"]
            bbox = BoxList(anno["boxes"], (width, height), mode="xyxy")
            bbox.add_field("labels", anno["labels"])
            bbox.add_field("difficult", anno["difficult"])
            record['bbox'] = bbox

        if 'mask' in self.data_type:
            mask = cv2.imread(self._mask_path % img_id, cv2.IMREAD_GRAYSCALE)
            record['mask'] = mask

        if self.transforms is not None:
            record = self.transforms(record)
            return record
        else:
            return None

    def __len__(self):
        return len(self.ids)

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._anno_path % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return PascalVOCDataset.CLASSES_NAMES[class_id]
