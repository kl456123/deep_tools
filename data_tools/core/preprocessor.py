# -*- coding: utf-8 -*-
"""
images: (N, )
labels: [xyxy, 1]
images_info: (N, 2)
"""

import numpy as np
import cv2
import os
import json
import time

from data_tools.utils.logger import setup_logger
from data_tools.utils.util import set_breakpoint, glob
from data_tools.core.transformer import Transformer
from data_tools.core.hdf5_converter import HDF5Converter


class Preprocessor(object):
    classes = [
        'bg', 'person', 'pet-cat', 'pet-dog', 'sofa', 'table',
        'bed', 'excrement', 'wire', 'key', 'shoes', 'socks', 'chair'
    ]
    classes_cn = [
        '背景', '人', '宠物-猫', '宠物-狗', '沙发', '桌子', '床', '粪便', '数据线', '钥匙', '鞋', '袜子', '椅子'
    ]

    def __init__(self, input_dirs,
                 output_dir,
                 input_size,
                 single_label=False,
                 ignore_error=False,
                 use_crop=True,
                 unknown_cls2bg=True):
        self.logger = setup_logger()
        if not isinstance(input_dirs, list):
            input_dirs = [input_dirs]
        for input_dir in input_dirs:
            if not os.path.exists(input_dir):
                self.logger.error('input_dir {} not exist'.format(input_dir))
                raise FileNotFoundError
        self.root_dir = input_dirs
        for input_dir in input_dirs:
            self.logger.info('input_dir: {}'.format(input_dir))
        self.logger.info('output_dir: {}'.format(output_dir))

        self.image_suffix = ['.JPG', '.jpg', '.JPEG', '.jpeg', '.png', '.PNG']
        self.label_suffix = ['.json', '.xml']
        self.ignore_error = ignore_error
        self.unknown_cls2bg = unknown_cls2bg

        self.images_path, self.labels_path = self.get_paths_pair()
        if not single_label:
            if not len(self.images_path) == len(self.labels_path):
                labels_path = ['{}{}'.format(
                    self.prefix_path(path), self.label_suffix[0]) for path in self.images_path]
                diff = set(labels_path)-set(self.labels_path)
                self.logger.debug(str(diff))
                if self.ignore_error:
                    self.consist_img_lbl()
                else:
                    raise RuntimeError

        # converter

        self.h5_converter = HDF5Converter(output_dir)
        self.transformer = Transformer(input_size, use_crop=use_crop)
        self.rgb = True

        self.class2id = self.generate_class2id_map(self.classes)
        self.class2id.update(self.generate_class2id_map(self.classes_cn))
        self.error_infos = []

        self.h5_converter.meta_data = self.create_meta_data()

    @staticmethod
    def prefix_path(path):
        return os.path.splitext(path)[0]

    def _spec_suffix(self, path, suffix):
        prefix_path = self.prefix_path(path)
        for s in suffix:
            if os.path.exists('{}{}'.format(prefix_path, s)):
                return s
        self.logger.error('No any suffix match the path: {}'.format(path))
        raise RuntimeError

    def consist_img_lbl(self):
        num_images = len(self.images_path)
        num_labels = len(self.labels_path)
        if num_images > num_labels:
            self.images_path = ['{}{}'.format(
                self.prefix_path(path), self._spec_suffix(path, self.image_suffix)) for path in self.labels_path]
        else:
            self.labels_path = ['{}{}'.format(
                self.prefix_path(path), self._spec_suffix(path, self.label_suffix)) for path in self.images_path]

    @staticmethod
    def generate_class2id_map(classes):
        class2id = {}
        for ind, class_name in enumerate(classes):
            class2id[class_name] = ind
        return class2id

    @staticmethod
    def generate_id2classes_map(classes):
        id2classes = {}
        for ind, class_name in enumerate(classes):
            id2classes[str(ind)] = class_name
        return id2classes

    def get_paths_pair(self):
        image_paths = []
        label_paths = []
        for input_dir in self.root_dir:
            image_paths.extend(glob(input_dir, self.image_suffix))

        for input_dir in self.root_dir:
            label_paths.extend(glob(input_dir, self.label_suffix))

        return sorted(image_paths), sorted(label_paths)

    def __len__(self):
        return len(self.images_path)

    def run_single(self, index):
        image_path = self.images_path[index]
        if len(self.labels_path) == 0:
            # if only single label file(e.g. coco)
            label_path = index
        else:
            label_path = self.labels_path[index]

        try:
            image = self.read_image(image_path)
        except:
            set_breakpoint()

        try:
            labels, success = self.read_labels(label_path)
        except:
            if self.ignore_error:
                success = False
            else:
                set_breakpoint()

        if not success:
            self.logger.error(
                'failed to read labels or no label exist in this image')
            self.error_infos.append(label_path)
            return

        samples = self.transformer(image, labels)
        if len(samples) == 0:
            self.logger.warning('samples is empty after transformed')
        for image, labels in samples:
            self.h5_converter.append(image, labels)

    def read_image(self, image_path):
        # using cv2 is better than PIL
        image = cv2.imread(image_path)
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for i in range(3):
                image[:, :, i] = cv2.equalizeHist(image[:, :, i])
        return image

    def read_labels(self, label_path):

        with open(label_path, 'r') as f:
            label = json.load(f)

        # handle corner case
        if 'result' not in label:
            label = {'result': label}

        if len(label['result']['data']) == 0:
            return None, False
        ih = label['result']['data'][0]['ih']
        iw = label['result']['data'][0]['iw']

        labels_info = []
        for ann in label['result']['data']:
            xs = []
            ys = []
            for point in ann['points']:
                xs.append(point['x'])
                ys.append(point['y'])
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            xmin = xs.min()
            ymin = ys.min()
            xmax = xs.max()
            ymax = ys.max()
            labels_info.append([
                xmin * iw, ymin * ih, xmax * iw, ymax * ih,
                float(self.class2id[ann['category'][0]])
            ])
        return np.asarray(labels_info).reshape(-1, 5), True

    def after_hook(self):
        self.h5_converter.close()
        if self.error_infos:
            self.logger.info(
                'failed nums/total nums({}/{})'.format(len(self.error_infos), len(self.images_path)))
            self.logger.error(str(self.error_infos))
        else:
            self.logger.info('No errors happened')

    def before_hook(self):
        pass

    def run(self, start_ind=0, end=None):
        self.logger.info('start processing ...')
        self.logger.info('Total num: {}'.format(len(self)))
        self.before_hook()
        if end is None:
            end = len(self)
        else:
            end = min(len(self), end)

        start_ind = min(len(self)-1, start_ind)
        for ind in range(start_ind,  end):
            self.run_single(ind)
            if ind and (ind+1) % 100 == 0:
                self.logger.info('{}/{}'.format(ind + 1, len(self)))

        self.after_hook()
        self.logger.info('end')

    def debug(self, index):
        self.logger.info('start debugging ...')
        self.before_hook()
        self.run_single(index)
        self.after_hook()

    @staticmethod
    def _get_time():
        return time.asctime(time.localtime(time.time()))

    def create_meta_data(self):
        meta_data = {}
        if self.rgb:
            meta_data['color_type'] = 'rgb'
        else:
            meta_data['color_type'] = 'bgr'

        meta_data['create_time'] = self._get_time()
        meta_data['version'] = '0.1.0'
        meta_data['max_num'] = self.h5_converter.max_num_samples_per_file
        return str(meta_data)


def main():
    dir_names = ['third_batch', 'fifth_batch',
                 'first_batch', 'second_batch', 'fourth_batch']
    # dir_names = ['']
    for dir_name in dir_names:
        input_dir = '/data/cleaner_machine/{}'.format(dir_name)
        output_dir = '/data/tmp2/{}'.format(dir_name)
        input_size = (320, 320)
        preprocessor = Preprocessor(
            input_dir, output_dir, input_size, ignore_error=True)
        preprocessor.run()
    # preprocessor.debug(595)


if __name__ == '__main__':
    main()
