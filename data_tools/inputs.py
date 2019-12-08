# -*- coding: utf-8 -*-
"""
images: (N, )
labels: [xyxy, 1]
images_info: (N, 2)
"""

import h5py
import numpy as np
import glob
import cv2
import os
import json
import logging
import sys


class HDF5Converter(object):
    KEY_IMAGES = 'images'
    KEY_LABELS_INFO = 'labels_info'
    KEY_IMAGES_INFO = 'images_info'

    # wh
    def __init__(self, h5_dir, size=(160, 160), max_num=1e8):
        self.h5_dir = h5_dir
        if not os.path.exists(self.h5_dir):
            os.makedirs(self.h5_dir)

        self.reset_buffer()
        self.input_size = size
        self.num_lbl_info_cols = 5
        self.num_img_info_cols = 2

        self.max_num_samples_per_file = max_num

        self._file_index = 0

    def reset_buffer(self):
        self.images = []
        self.labels_info = []
        self.images_info = []

    def generate_h5_path(self):
        h5_path = os.path.join(self.h5_dir, '{:03d}.h5'.format(
            self._file_index))
        self._file_index += 1
        return h5_path

    @property
    def full(self):
        return len(self.images) == self.max_num_samples_per_file

    @property
    def empty(self):
        return len(self.images)==0

    def append(self, image, labels):
        # wh
        input_size = self.input_size
        # resize for image
        #(hwc)
        fx = input_size[0] / image.shape[1]
        fy = input_size[1] / image.shape[0]

        image = cv2.resize(image, (int(input_size[0]), int(input_size[1])))
        self.images.append(image.flatten())
        self.images_info.append(np.asarray(image.shape[:2]))

        # resize for labels
        labels = np.asarray(labels)
        labels[:, 0] = labels[:, 0] * fx
        labels[:, 1] = labels[:, 1] * fy
        labels[:, 2] = labels[:, 2] * fx
        labels[:, 3] = labels[:, 3] * fy

        self.labels_info.append(labels.flatten())

        if self.full:
            self.save()

    def save(self):
        h5_path = self.generate_h5_path()
        num_images = len(self.images)

        h5_db = h5py.File(h5_path, mode='w')
        labels_info_dt = h5py.special_dtype(vlen=np.float32)
        image_dt = h5py.special_dtype(vlen=np.uint8)
        h5_db.create_dataset(
            self.KEY_IMAGES, (num_images, ), dtype=image_dt)
        h5_db.create_dataset(
            self.KEY_LABELS_INFO, (num_images, ),
            dtype=labels_info_dt)
        h5_db.create_dataset(
            self.KEY_IMAGES_INFO, (num_images, self.num_img_info_cols),
            dtype=np.int32)

        h5_db[self.KEY_IMAGES][...] = self.images

        h5_db[self.KEY_LABELS_INFO][...] = np.asarray(self.labels_info)
        h5_db[self.KEY_IMAGES_INFO][...] = np.asarray(self.images_info)

        h5_db.close()
        # clear buffer
        self.reset_buffer()

    def close(self):
        if not self.empty:
            self.save()

    @classmethod
    def load(cls, h5_dir):
        files = sorted(glob.glob(os.path.join(h5_dir, '*.h5')))
        # merge alll h5 files
        images = []
        images_info = []
        labels_info = []
        for file in files:
            h5_db = h5py.File(file, mode='r')
            images.append(h5_db[cls.KEY_IMAGES])
            images_info.append(h5_db[cls.KEY_IMAGES_INFO])
            labels_info.append(h5_db[cls.KEY_LABELS_INFO])
        images = np.concatenate(images)
        images_info = np.concatenate(images_info)
        labels_info = np.concatenate(labels_info)
        return images, images_info, labels_info


class Preprocessor(object):
    classes = [
            '__background__', 'person', 'pet-cat', 'pet-dog', 'sofa', 'table',
            'bed', 'excrement', 'wire', 'key'
        ]
    classes_cn = [
            '背景', '人', '宠物-猫', '宠物-狗', '沙发', '桌子', '床', '粪便', '数据线', '钥匙'
        ]
    def __init__(self, input_dir, output_dir, input_size):
        self.root_dir = input_dir

        self.logger = logging.getLogger('preprocessor')
        self.logger.setLevel(logging.INFO)

        self.image_suffix = ['.JPG', '.jpg', '.JPEG', '.jpeg', '.png', '.PNG']
        self.label_suffix = ['.json']

        self.images_path, self.labels_path = self.get_paths_pair()
        assert len(self.images_path) == len(self.labels_path)

        # converter

        self.h5_converter = HDF5Converter(output_dir, input_size)
        self.rgb = True

        self.class2id = self.generate_class2id_map(self.classes)
        self.class2id.update(self.generate_class2id_map(self.classes_cn))

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
        return sorted(self.glob(self.root_dir, self.image_suffix)), sorted(
            self.glob(self.root_dir, self.label_suffix))

    @staticmethod
    def glob(root_dir, suffix=[]):
        files = []
        for s in suffix:
            files.extend(
                glob.glob(
                    os.path.join(root_dir, '**/*{}'.format(s)),
                    recursive=True))
        return files

    def __len__(self):
        return len(self.images_path)

    def run_single(self, index):
        image_path = self.images_path[index]
        label_path = self.labels_path[index]

        image = self.read_image(image_path)
        labels, success = self.read_labels(label_path)
        if success:
            self.h5_converter.append(image, labels)
        else:
            self.logger.warn(
                'failed to read labels or no label exist in this image')

    def read_image(self, image_path):
        # using cv2 is better than PIL
        image = cv2.imread(image_path)
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    def before_hook(self):
        pass

    def run(self):
        self.logger.info('start ...')
        self.logger.info('Total num: {}'.format(len(self)))
        self.before_hook()
        for ind in range(len(self)):
            self.run_single(ind)
            sys.stdout.write('\r{}/{}'.format(ind + 1, len(self)))
            sys.stdout.flush()

        self.after_hook()
        self.logger.info('end')


def main():
    input_dir = '/data/cleaner_machine/first_batch'
    output_dir = './tmp'
    input_size = (160, 160)
    preprocessor = Preprocessor(input_dir, output_dir, input_size)
    preprocessor.run()


if __name__ == '__main__':
    main()
