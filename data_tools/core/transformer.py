# -*- coding: utf-8 -*-
import numpy as np
import cv2


class Transformer(object):
    def __init__(self, input_size, use_crop=True):
        self.crop_list = [0.5] * 10
        self.spec_classes = [9, 8, 7]
        self.min_box_size = 8

        # the smallest objects should occupy some place
        self.min_ratio = 30.0/320.0
        self.input_size = input_size
        self.use_crop = use_crop

    def __call__(self, *sample):
        # may be generate multiple samples
        if self.use_crop:
            samples = self.crop(*sample)
        else:
            samples = [sample]
        resized_samples = []
        for sample in samples:
            resized_sample = self.resize(*sample)
            image, labels = self.size_filter(*resized_sample)
            if labels.shape[0] > 0:
                resized_samples.append((image, labels))

        return resized_samples

    def crop(self, image, labels_info):
        samples = []

        # import ipdb
        # ipdb.set_trace()
        # crop for small object
        image1, labels_info1, no_used = self._crop(
            image, labels_info.copy())
        if labels_info1.shape[0] > 0:
            samples.append((image1, labels_info1))

        # if no_used.any():
            # # append directly
            # samples.append((image, labels_info[no_used]))
        return samples

    def _crop(self, image, labels_info, ):
        image_shape = image.shape
        window, success = self._calc_window(
            labels_info, image_shape)

        if not success:
            keep = np.zeros(labels_info.shape[0]).astype(np.bool)
            return image, labels_info[keep], np.logical_not(keep)

        # crop image
        x1, y1, x2, y2 = window.flatten()
        cropped_image = image[y1:y2+1, x1:x2+1]

        # crop label
        labels_info[:, 0] = np.minimum(labels_info[:, 0], x2)-x1
        labels_info[:, 2] = np.minimum(labels_info[:, 2], x2)-x1
        labels_info[:, 1] = np.minimum(labels_info[:, 1], y2)-y1
        labels_info[:, 3] = np.minimum(labels_info[:, 3], y2)-y1
        labels_info[:, :4] = np.maximum(labels_info[:, :4], 0)

        # filter invalid boxes
        # out of window
        wh = labels_info[:, 2:4]-labels_info[:, :2]
        # too small
        area_keep = wh[:, 0]*wh[:, 1] > self.min_box_size

        keep = area_keep
        # no any box
        if not keep.any():
            return image, labels_info[keep], np.logical_not(keep)

        labels_info = labels_info[keep]

        return cropped_image, labels_info, np.logical_not(keep)

    def _calc_wh(self, spec_boxes):
        spec_boxes_wh = spec_boxes[:, 2:] - spec_boxes[:, :2]
        # min_w, min_h = spec_boxes_wh[:, 0].min(), spec_boxes_wh[:, 1].min()
        min_box_scale = np.sqrt(
            spec_boxes_wh[:, 0] * spec_boxes_wh[:, 1]).min()
        wh = min_box_scale/self.min_ratio
        return wh

    def _calc_window(self, labels_info, image_shape):
        # calc crop window
        boxes = labels_info[:, :4]
        classes = labels_info[:, 4]
        keep = np.ones_like(classes).astype(np.bool)
        for spec_class in self.spec_classes:
            keep = np.logical_or(keep, classes == spec_class)

        if not keep.any():
            return None, False
        # calc xy
        spec_boxes = boxes[keep]
        xmin = spec_boxes[:, 0].min()
        ymin = spec_boxes[:, 1].min()
        xmax = spec_boxes[:, 2].max()
        ymax = spec_boxes[:, 3].max()
        window = np.asarray([xmin, ymin, xmax, ymax])
        xy = (window[2:]+window[:2])/2

        # calc wh
        wh = self._calc_wh(boxes)

        window = np.concatenate([xy-wh*0.5, xy+wh*0.5], axis=0)

        # check window
        x1, y1, x2, y2 = window.flatten()
        img_h, img_w = image_shape[:2]
        x1 = min(max(x1, 0), img_w-1)
        y1 = min(max(y1, 0), img_h-1)
        x2 = min(max(x2, 0), img_w-1)
        y2 = min(max(y2, 0), img_h-1)
        if x2-x1 == 0 or y2-y1 == 0:
            success = False
        else:
            success = True
        window = np.ceil(np.asarray([x1, y1, x2, y2])).astype(np.int)

        return window, success

    def resize(self, image, labels):
        # wh
        input_size = self.input_size

        # then resize for image
        fx = input_size[0] / image.shape[1]
        fy = input_size[1] / image.shape[0]

        image = cv2.resize(image, (int(input_size[0]), int(input_size[1])))

        # resize for labels
        labels[:, 0] = labels[:, 0] * fx
        labels[:, 1] = labels[:, 1] * fy
        labels[:, 2] = labels[:, 2] * fx
        labels[:, 3] = labels[:, 3] * fy

        return image, labels

    def size_filter(self, image, labels):
        wh = labels[:, 2:4] - labels[:, :2]
        size_cond = np.logical_and(
            wh[:, 0] > self.min_box_size, wh[:, 1] > self.min_box_size)
        # ignore size filter for bg
        fg_cond = labels[:, -1] > 0

        cond = ~fg_cond | (fg_cond & size_cond)

        labels = labels[cond]
        return image, labels
