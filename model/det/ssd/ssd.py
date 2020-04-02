from torch import nn, stack
from model.ssd.backbone import build_backbone
from model.ssd.box_head import build_box_head
from model.ssd.utils.prior_matcher import PriorMatcher
from model.ssd.anchors.prior_box import PriorBox


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)
        self.prior_matcher = PriorMatcher(PriorBox(cfg)(),
                                          cfg.MODEL.CENTER_VARIANCE,
                                          cfg.MODEL.SIZE_VARIANCE,
                                          cfg.MODEL.THRESHOLD)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses, detections
        return detections


class NetWrapper(nn.Module):
    def __init__(self, net):
        super(NetWrapper, self).__init__()
        self.net = net

    def forward(self, record):
        image = record['image']
        if self.training:
            bbox = record['bbox']
            boxes_list = []
            labels_list = []
            for i in range(len(bbox)):
                boxes, labels = self.net.prior_matcher(bbox[i].get_percent_coords(), bbox[i].get_field('labels'))
                boxes_list.append(boxes)
                labels_list.append(labels)

            targets = {'boxes': stack(boxes_list), 'labels': stack(labels_list)}
            losses, result = self.net(image, targets)
            total_loss = sum(loss for loss in losses.values())
            loss = {'total_loss': total_loss, 'reg_loss': losses['reg_loss'], 'cls_loss': losses['cls_loss']}
            prediction = (result,)
            return loss, prediction
        else:
            prediction = self.net(image)
            return prediction
