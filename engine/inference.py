# Author: Xiaoyang Wu (gofinge@foxmail.com)
import logging
import torch
import numpy as np

from data.structures.bounding_box import bbox_list_to_np
from torch.nn.parallel import DistributedDataParallel
from utils.miscellaneous import TqdmBar, move_to_device
from utils.comm import get_rank
from utils.metric_logger import MetricLogger
from utils.summary import write_summary
from configs.defaults import INFERENCE


class MeanOfAveragePrecision(object):
    def __init__(self, num_class, max_objs=50, fppi=(0.1, )):
        self.max_objs = max_objs
        self.num_class = num_class
        self.object_numbers = np.zeros(self.num_class, dtype=np.int64)
        self.object_overlaps = []
        self.iou_range = np.arange(0.5, 1.0, 0.05)
        self.gt_mask = []
        self.batch_count = 0
        self.fppi = fppi

    def calculate_overlaps(self, record, prediction):
        # prediction results, dets = [nbatch, K, x1 + y1 + x2 + y2 + score + class]
        # reshape to [nbatch * K, x1 + y1 + x2 + y2 + score + class]
        dets = np.reshape(np.squeeze(prediction.numpy()), (-1, 6))

        bbox = record["bbox"]
        bbox_gt = bbox_list_to_np(bbox, max_objs=self.max_objs)

        self.object_numbers += np.asarray([len(np.where(bbox_gt[:, 5] == i)[0]) for i in range(self.num_class)])
        reg_mask = np.zeros(len(prediction) * self.max_objs, dtype=np.float)
        reg_mask[np.where(bbox_gt[:, 4] == 1)] = True
        K = self.max_objs

        for i in range(len(dets)):
            det = dets[i]
            bbox_mask = np.ones(len(bbox_gt)).astype(np.bool)
            bbox_mask[int(i / K) * K:int(i / K + 1) * K] = False
            bbox_mask[np.where(bbox_gt[:, 5] != det[5])] = True
            invert_reg_mask = np.logical_not(reg_mask)
            bbox_mask[invert_reg_mask] = True

            ixmin = np.maximum(bbox_gt[:, 0], det[0])
            iymin = np.maximum(bbox_gt[:, 1], det[1])
            ixmax = np.minimum(bbox_gt[:, 2], det[2])
            iymax = np.minimum(bbox_gt[:, 3], det[3])

            iw = np.maximum(ixmax - ixmin + 1, 0.)
            ih = np.maximum(iymax - iymin + 1, 0.)
            inters = iw * ih
            union = (det[2] - det[0] + 1.) * (det[3] - det[1] + 1.) - inters + \
                    (bbox_gt[:, 2] - bbox_gt[:, 0] + 1.) * (bbox_gt[:, 3] - bbox_gt[:, 1] + 1.)
            overlaps = inters / union
            overlaps[bbox_mask] = 0

            overlap_max = np.max(overlaps)
            object_index = np.argmax(overlaps) if overlap_max else -1

            # res = [confidence, max IoU, gt bounding box index, batch count, category, TP, FP]
            res = np.zeros(7)
            res[0] = det[4]
            res[1] = overlap_max
            res[2] = object_index
            res[3] = self.batch_count
            res[4] = det[5]
            self.object_overlaps.append(res)
        self.batch_count += 1
        self.gt_mask.append(reg_mask)

    def calculate_map(self):
        if len(self.object_overlaps) == 0:
            return 0, 0

        # self.object_overlaps = [number,
        # confidence(0) + max IoU(1) + gt bounding box index(2) + batch count(3) + category(4) + TP(5) + FP(6)]
        object_overlaps = np.asarray(self.object_overlaps, dtype=np.float32)
        sorted_ind = np.argsort(-object_overlaps[:, 0])
        object_overlaps = object_overlaps[sorted_ind, :]
        object_overlaps = np.tile(object_overlaps[:, :, np.newaxis], (1, 1, len(self.iou_range)))
        object_overlaps[:, 4, :].astype(np.int64)
        gt_mask = np.asarray(self.gt_mask, dtype=np.int32)
        gt_mask = np.tile(gt_mask[np.newaxis, :], (len(self.iou_range), 1, 1))

        for object_overlap in object_overlaps:
            confidence, iou_max, object_index, batch_count = \
                object_overlap[0, 0], object_overlap[1, 0], int(object_overlap[2, 0]), int(object_overlap[3, 0])
            if object_index == -1:
                object_overlap[6, :] = 1
                continue

            tp_mask = np.zeros(len(gt_mask)).astype(np.bool)
            tp_mask[np.where(iou_max > self.iou_range)] = True
            tp_mask[np.where(gt_mask[:, batch_count, object_index] != 1)] = False
            fp_mask = np.logical_not(tp_mask)

            gt_mask[tp_mask, batch_count, object_index] = 0
            object_overlap[5, tp_mask] = 1
            object_overlap[6, fp_mask] = 1

        mAP_50_95 = []
        mAP_50 = []
        recall_fppi = []
        object_overlaps = object_overlaps.swapaxes(1, 2)
        category = object_overlaps[:, 0, 4]

        # ignore label 0
        for cls in range(1, self.num_class):
            cls_ind = np.where(category == cls)
            if not len(cls_ind[0]):
                continue
            tp = np.squeeze(object_overlaps[cls_ind, :, 5], axis=0)
            fp = np.squeeze(object_overlaps[cls_ind, :, 6], axis=0)
            acc_tp = np.cumsum(tp, axis=0)
            acc_fp = np.cumsum(fp, axis=0)

            precision = acc_tp / np.maximum(acc_tp + acc_fp, np.finfo(np.float64).eps)
            recall = acc_tp / np.maximum(self.object_numbers[cls], np.finfo(np.float64).eps)
            precision = precision.T  # precision = [iou_range, nums]
            recall = recall.T

            # fppi filter, only use iou threshold = 0.5
            mask_shape = (len(self.fppi),) + acc_tp.shape
            fppi_mask = np.zeros(mask_shape, dtype=np.int)
            for i, fppi in enumerate(self.fppi):
                fppi_mask[i][np.where(acc_fp < self.object_numbers[cls] * fppi)] = 1
            fppi_mask = np.sum(fppi_mask, axis=1)
            fppi_mask -= np.ones(fppi_mask.shape, dtype=np.int)
            fppi_mask[np.where(fppi_mask < 0)] = 0
            recall_fppi.append([recall[0][fppi[0]] for fppi in fppi_mask])

            AP_iou = []
            for prec, rec in zip(precision, recall):
                mrec = np.concatenate(([0.], rec, [1.]))
                mpre = np.concatenate(([0.], prec, [0.]))

                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])
                AP_iou.append(ap)

            mAP_50_95.append(sum(AP_iou) / (len(AP_iou)))
            mAP_50.append(AP_iou[0])
        average_mAP_50_95 = sum(mAP_50_95) / max(len(mAP_50_95), np.finfo(np.float64).eps)
        average_mAP_50 = sum(mAP_50) / max(len(mAP_50), np.finfo(np.float64).eps)
        if not len(recall_fppi):
            average_recall_fppi = [0 for _ in range(len(self.fppi))]
        else:
            average_recall_fppi = np.sum(recall_fppi, axis=0) / max(len(recall_fppi), np.finfo(np.float64).eps)

        return average_mAP_50_95, average_mAP_50, average_recall_fppi


class MeanOfIntersectionOverUnion(object):
    def __init__(self, num_class, ignore_index=255):
        self.intersection = np.zeros(num_class)
        self.union = np.zeros(num_class)
        self.num_class = num_class
        self.ignore_index = ignore_index

    def calculate_miou(self, record, prediction):
        assert "mask" in record
        label = record["mask"]
        label = label.detach().numpy()
        if len(prediction.shape) == 2:
            pred = np.array([prediction])
        else:
            pred = prediction

        pred = pred.reshape(pred.size).copy()
        label = label.reshape(label.size)

        pred[np.where(label == self.ignore_index)[0]] = self.ignore_index
        intersection = pred[np.where(pred == label)[0]]

        area_intersection, _ = np.histogram(intersection, bins=np.arange(self.num_class + 1))
        area_output, _ = np.histogram(pred, bins=np.arange(self.num_class + 1))
        area_target, _ = np.histogram(label, bins=np.arange(self.num_class + 1))
        area_union = area_output + area_target - area_intersection
        self.intersection += area_intersection
        self.union += area_union

    def calculate_batch_miou(self):
        return np.mean(self.intersection / (self.union + 1e-10))


@INFERENCE.register("SSD")
def detection_inference(cfg, model, data_loader_val, device, iteration, summary_writer=None, logger=None,
                        visualize=False, fppi=(0.1, 0.01)):
    mAP = MeanOfAveragePrecision(cfg.DATASET.NUM_CLASS, cfg.DATASET.MAX_OBJECTS, fppi=fppi)
    bar = TqdmBar(data_loader_val, 0, get_rank(), data_loader_val.__len__(),
                  description='Inference', use_bar=cfg.USE_BAR)
    for iteration, record in bar.bar:
        record = move_to_device(record, device)

        prediction = model(record)

        prediction = prediction.cpu().detach()
        record = move_to_device(record, torch.device('cpu'))

        mAP.calculate_overlaps(record, prediction)
        if visualize:
            # TODO vis module
            pass

    mAP_5095, mAP_50, m_recall = mAP.calculate_map()

    if logger is not None:
        logger.info('====================================================================================')
        logger.info('Average inference time per image without post process is: %s' % (
                sum(model.inference_time_without_postprocess) / max(len(model.inference_time_without_postprocess),
                                                                    np.finfo(np.float64).eps)))
        logger.info('Average inference time per image with post process is: %s' % (
                sum(model.inference_time_with_postprocess) / max(len(model.inference_time_with_postprocess),
                                                                 np.finfo(np.float64).eps)))

        logger.info('mAP(@iou=0.5:0.95): %s' % mAP_5095)
        logger.info('mAP(@iou=0.5): %s' % mAP_50)
        logger.info('Recall(@iou=0.5, @fppi=%s): %s' % (fppi[0], m_recall[0]))
        logger.info('Recall(@iou=0.5, @fppi=%s): %s' % (fppi[1], m_recall[1]))
        logger.info('====================================================================================')
    if summary_writer is not None:
        record = {'mAP_iou_0.5_0.95': mAP_5095, 'mAP_iou_0.5': mAP_50,
                  'Recall_iou_0.5_fppi_{}'.format(fppi[0]): m_recall[0],
                  'Recall_iou_0.5_fppi_{}'.format(fppi[1]): m_recall[1]}
        write_summary(summary_writer, iteration, record=record, group='Evaluations')


def build_inference(cfg):
    if cfg.MODEL.NAME.upper() in INFERENCE.keys():
        return INFERENCE[cfg.MODEL.NAME.upper()]
    else:
        raise ValueError('Unknown model type! Please confirm your config file.')


@torch.no_grad()
def do_evaluation(cfg, model, data_loader_val, device, arguments, summary_writer):
    # get logger
    logger = logging.getLogger(cfg.NAME)
    logger.info("Start evaluation  ...")
    if isinstance(model, DistributedDataParallel):
        model = model.module
    model.eval()
    meters_val = MetricLogger(delimiter="  ")
    # TODO: add compare module which can test sampled train set
    # bar = TqdmBar(data_loader_val, 0, get_rank(), data_loader_val.__len__(),
    #               description='Validation', use_bar=cfg.USE_BAR)
    # for iteration, record in bar.bar:
    #     record = move_to_device(record, device)
    #     loss, prediction = model(record)
    #     # reduce losses over all GPUs for logging purposes
    #     loss_reduced = {key: value.cpu().item() for key, value in loss.items()}
    #     meters_val.update(**loss_reduced)
    # bar.close()
    # logger.info(
    #     meters_val.delimiter.join(
    #         [
    #             "[Validation]: ",
    #             "iter: {iter}",
    #             "{meters}",
    #             "mem: {memory:.0f}",
    #         ]
    #     ).format(
    #         iter=arguments["iteration"],
    #         meters=str(meters_val),
    #         memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
    #     )
    # )
    record = {name: meter.global_avg for name, meter in meters_val.meters}
    write_summary(summary_writer, arguments["iteration"], record=record, group='Valid_Losses')

    inference = build_inference(cfg)
    model.eval()
    inference(cfg, model, data_loader_val, device, iteration=arguments["iteration"], summary_writer=summary_writer,
              logger=logger, visualize=False)
    model.train()




