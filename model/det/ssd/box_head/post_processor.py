import torch

from model.det.ssd.utils.nms import batched_nms


class PostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.DIMS[0]
        self.height = cfg.INPUT.DIMS[1]

    def __call__(self, detections):
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        bbox_tensor = torch.zeros(batch_size, self.cfg.DATASET.MAX_OBJECTS, 6)
        for batch_id in range(batch_size):
            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]

            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            indices = torch.nonzero(scores > self.cfg.TEST.CONFIDENCE_THRESHOLD).squeeze(1)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height

            keep = batched_nms(boxes, scores, labels, self.cfg.TEST.NMS_THRESHOLD)
            # keep only topk scoring predictions
            keep = keep[:self.cfg.DATASET.MAX_OBJECTS]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            valid_objs = int(labels.shape[0])
            bbox_tensor[batch_id, :valid_objs, :4] = boxes
            bbox_tensor[batch_id, :valid_objs, 4] = scores
            bbox_tensor[batch_id, :valid_objs, 5] = labels

        return bbox_tensor
