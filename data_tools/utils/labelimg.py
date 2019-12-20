# -*- coding: utf-8 -*-

import os
import cv2
from data_tools.utils.logger import setup_logger
from data_tools.utils.util import glob
import json


def label_img():
    pass


class AnnotationTools(object):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.annos = []
        self.logger = setup_logger()
        self.store_name = 'anno.json'
        self.image_suffixs = ['.jpg', '.png']

    def read_image(self, image_path):
        return cv2.imread(image_path)

    def read_dir(self):
        return glob(self.image_dir, self.image_suffixs)

    def save(self):
        results = {'anno': self.annos, 'version': '0.1.0'}
        json.dump(results, self.store_name)

    def annotate(self, image):
        flag = True
        while flag:
            init_rect = cv2.selectROI('ann', image, False, False)
            try:
                class_id = int(input("input class index: "))
            except:
                flag = False
                continue
            anno = []
            anno.extend(init_rect)
            anno.append(class_id)
            self.annos.append(anno)

    def run(self):
        image_paths = self.read_dir()
        for image_path in image_paths:
            image = self.read_image(image_path)
            self.annotate(image)


def main():
    image_dir = '/data/test_images'
    annotation_tools = AnnotationTools(image_dir)
    annotation_tools.run()


if __name__ == '__main__':
    main()
