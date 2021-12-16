import os
import torch
import cv2 as cv
import numpy as np
from sort.sort import Sort
from models import experimental as exp
from utils import general, datasets
from deepsort.utils.parser import get_config
from deepsort.deep_sort import DeepSort
from utils.metrics import bbox_iou
from yolov3.pytorchyolo import models, detect


def is_not_intersect(cur_bbox, bbox):
    return bbox[2] < cur_bbox[0] or bbox[0] > cur_bbox[2] or bbox[1] > cur_bbox[3] or bbox[3] < cur_bbox[1]


class YoloV3Detector:
    def __init__(self, conf_thres=0.2, iou_thres=0.2, mode='deep_sort'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.load_model("yolov3/config/yolov3.cfg", "yolov3/weights/yolov3.weights")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.mode = mode
        config = get_config()
        config.merge_from_file("deepsort/configs/deep_sort.yaml")
        self.tracker = None
        if self.mode == 'deep_sort':
            self.tracker = DeepSort(config.DEEPSORT.REID_CKPT,
                                       max_dist=config.DEEPSORT.MAX_DIST,
                                       min_confidence=config.DEEPSORT.MIN_CONFIDENCE,
                                       max_iou_distance=config.DEEPSORT.MAX_IOU_DISTANCE,
                                       max_age=config.DEEPSORT.MAX_AGE,
                                       n_init=config.DEEPSORT.N_INIT,
                                       nn_budget=config.DEEPSORT.NN_BUDGET,
                                       use_cuda=True)
        elif self.mode == 'sort':
            self.tracker = Sort()
        elif self.mode == 'iou':
            self.id = 1
            self.cur_bboxes = None
            self.skipped_bboxes = {}

    def detect(self, input_img):
        img0 = input_img.copy()
        img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
        prediction = detect.detect_image(self.model, img)
        bbox_xywh = []
        confidences = []
        classes = []
        if len(prediction) > 0:
            # x1 y1 x2 y2 conf lbl
            if self.mode == 'deep_sort':
                for x1, y1, x2, y2, conf, lbl in prediction:
                    label = int(lbl)
                    if label in [0, 1, 2, 3, 5, 7]:
                        bbox_xywh.append([0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1])
                        confidences.append(conf)
                        classes.append(label)
                bbox_xywh = torch.Tensor(bbox_xywh)
                confidences = torch.Tensor(confidences)
                classes = torch.Tensor(classes)
                if len(bbox_xywh):
                    deep_sort_output = self.tracker.update(bbox_xywh, confidences, classes, img0)
                    return deep_sort_output

            elif self.mode == 'iou':
                if self.cur_bboxes is None:
                    self.cur_bboxes = {}
                    for x1, y1, x2, y2, conf, lbl in prediction:
                        label = int(lbl)
                        if label in [0, 1, 2, 3, 5, 7]:
                            self.cur_bboxes[(x1, y1, x2, y2)] = [self.id, label]
                            self.id += 1
                    return []
                else:
                    new_bboxes = {}
                    for x1, y1, x2, y2, conf, lbl in prediction:
                        label = int(lbl)
                        if label in [0, 1, 2, 3, 5, 7]:
                            bbox = (x1, y1, x2, y2)
                            if bbox in self.cur_bboxes:
                                new_bboxes[bbox] = self.cur_bboxes[bbox]
                                self.cur_bboxes.pop(bbox)
                            else:
                                match_found = False
                                max_iou = 0
                                best_box = None
                                for cur_bbox in list(self.cur_bboxes):
                                    if is_not_intersect(cur_bbox, bbox):
                                        continue
                                    iou = bbox_iou(torch.Tensor(np.asarray(bbox)), torch.Tensor(np.asarray(cur_bbox)))

                                    if iou > max_iou:
                                        max_iou = iou
                                        best_box = cur_bbox
                                if max_iou > 0.3:
                                    new_bboxes[bbox] = self.cur_bboxes[best_box]
                                    self.cur_bboxes.pop(best_box)
                                    match_found = True

                                if not match_found:
                                    for skip_bbox in list(self.skipped_bboxes):
                                        if is_not_intersect(skip_bbox, bbox):
                                            continue
                                        iou = bbox_iou(torch.Tensor(np.asarray(bbox)), torch.Tensor(np.asarray(skip_bbox)))
                                        if float(iou) > 0.3:
                                            new_bboxes[bbox] = self.skipped_bboxes[skip_bbox]
                                            self.skipped_bboxes.pop(skip_bbox)
                                            break
                                if bbox not in new_bboxes:
                                    new_bboxes[bbox] = [self.id, label]
                                    self.id += 1

                    self.cur_bboxes = new_bboxes
                    output = []
                    for bbox in list(self.cur_bboxes):
                        tup = self.cur_bboxes[bbox]
                        output.append(list(bbox) + tup)
                    return output
            elif self.mode == 'sort':
                dets = []

                sort_output = []
                for x1, y1, x2, y2, conf, lbl in prediction:
                    label = int(lbl)
                    if label in [0, 1, 2, 3, 5, 7]:
                        dets.append([x1, y1, x2, y2, conf])
                detections = self.tracker.update(np.asarray(dets))
                for bbox in detections:
                    sort_output.append(list(bbox) + [0])
                return sort_output
        return []


class YoloV5Detector:
    def __init__(self, model_name='yolov5l', size=640, half=True, conf_thres=0.2, iou_thres=0.2, mode='deep_sort'):
        self.weights = os.path.join("weights/", model_name + ".pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.model = exp.attempt_load(self.weights, map_location=self.device)
        self.half = half
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model.to(self.device).eval()
        self.stride = int(self.model.stride.max())
        if self.half:
            self.model.half()
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        config = get_config()
        config.merge_from_file("deepsort/configs/deep_sort.yaml")
        self.ds_tracker = DeepSort(config.DEEPSORT.REID_CKPT,
                                   max_dist=config.DEEPSORT.MAX_DIST,
                                   min_confidence=config.DEEPSORT.MIN_CONFIDENCE,
                                   max_iou_distance=config.DEEPSORT.MAX_IOU_DISTANCE,
                                   max_age=config.DEEPSORT.MAX_AGE,
                                   n_init=config.DEEPSORT.N_INIT,
                                   nn_budget=config.DEEPSORT.NN_BUDGET,
                                   use_cuda=True)
        self.mode = mode
        if self.mode == 'iou':
            self.id = 1
            self.cur_bboxes = None
            self.skipped_bboxes = {}

    def load_img(self, img):
        img0 = img.copy()
        img = datasets.letterbox(img, new_shape=self.size, stride=self.stride, scaleup=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
        if self.half:
            img = img.half()
        img = img / 255.0
        img = img.unsqueeze(0)
        return img0, img

    def detect(self, input_img):
        img0, img = self.load_img(input_img)
        prediction = self.model(img, augment=False)[0]
        prediction = general.non_max_suppression(prediction.float(), self.conf_thres, self.iou_thres)[0]
        bbox_xywh = []
        confidences = []
        classes = []
        if len(prediction) > 0:
            # x1 y1 x2 y2 conf lbl
            prediction[:, :4] = general.scale_coords(img.shape[2:], prediction[:, :4], img0.shape).round()
            if self.mode == 'deep_sort':
                for *coord, confidence, cls in prediction:
                    label = int(cls)
                    if label in [0, 1, 2, 3, 5, 7]:
                        x1 = int(coord[0])
                        y1 = int(coord[1])
                        x2 = int(coord[2])
                        y2 = int(coord[3])
                        conf = float(confidence)
                        bbox_xywh.append([int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2)), x2 - x1, y2 - y1])
                        confidences.append(conf)
                        classes.append(label)
                bbox_xywh = torch.Tensor(bbox_xywh)
                confidences = torch.Tensor(confidences)
                classes = torch.Tensor(classes)
                if len(bbox_xywh):
                    deep_sort_output = self.ds_tracker.update(bbox_xywh, confidences, classes, img0)
                    return deep_sort_output
            elif self.mode == 'iou':
                if self.cur_bboxes is None:
                    self.cur_bboxes = {}
                    for *coord, _, cls in prediction:
                        label = int(cls)
                        if label in [0, 1, 2, 3, 5, 7]:
                            x1 = int(coord[0])
                            y1 = int(coord[1])
                            x2 = int(coord[2])
                            y2 = int(coord[3])
                            self.cur_bboxes[(x1, y1, x2, y2)] = [self.id, label]
                            self.id += 1
                    return []
                else:
                    new_bboxes = {}
                    for *coord, _, cls in prediction:
                        label = int(cls)
                        if label in [0, 1, 2, 3, 5, 7]:
                            x1 = int(coord[0])
                            y1 = int(coord[1])
                            x2 = int(coord[2])
                            y2 = int(coord[3])
                            bbox = (x1, y1, x2, y2)
                            if bbox in self.cur_bboxes:
                                new_bboxes[bbox] = self.cur_bboxes[bbox]
                                self.cur_bboxes.pop(bbox)
                            else:
                                match_found = False
                                max_iou = 0
                                best_box = None
                                for cur_bbox in list(self.cur_bboxes):
                                    if is_not_intersect(cur_bbox, bbox):
                                        continue
                                    iou = bbox_iou(torch.Tensor(np.asarray(bbox)), torch.Tensor(np.asarray(cur_bbox)))

                                    if iou > max_iou:
                                        max_iou = iou
                                        best_box = cur_bbox
                                if max_iou > 0.3:
                                    new_bboxes[bbox] = self.cur_bboxes[best_box]
                                    self.cur_bboxes.pop(best_box)
                                    match_found = True

                                if not match_found:
                                    for skip_bbox in list(self.skipped_bboxes):
                                        if is_not_intersect(skip_bbox, bbox):
                                            continue
                                        iou = bbox_iou(torch.Tensor(np.asarray(bbox)),
                                                       torch.Tensor(np.asarray(skip_bbox)))
                                        if float(iou) > 0.3:
                                            new_bboxes[bbox] = self.skipped_bboxes[skip_bbox]
                                            self.skipped_bboxes.pop(skip_bbox)
                                            break
                                if bbox not in new_bboxes:
                                    new_bboxes[bbox] = [self.id, label]
                                    self.id += 1
                    self.cur_bboxes = new_bboxes
                    output = []
                    for bbox in list(self.cur_bboxes):
                        tup = self.cur_bboxes[bbox]
                        output.append(list(bbox) + tup)
                    return output
            return []
