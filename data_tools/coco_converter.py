# -*- coding: utf-8 -*-

"""

info:
image: file_name, height, width
annotations: xmin, ymin, xmax, ymax, name

"""
import json
from collections import OrderedDict

class COCOConverter(object):
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []

        # ids
        self.image_id = 0
        self.anno_id = 0
        self.cat_id = 0

        self.cat2id_map = {}

    def cat2id(self, class_name):
        if class_name not in self.cat2id_map:
            self.cat2id_map[class_name] = self.cat_id
            self.append_categories(class_name)
        return self.cat2id_map[class_name]

    def append_images(self, info):
        image = OrderedDict()
        image['file_name'] = info['file_name']
        image['height'] = info['height']
        image['width'] = info['width']
        image['id'] = self.image_id
        self.images.append(image)

    def append_annotations(self, annotations):
        for info in annotations:

            x1 = int(info['xmin'])
            y1 = int(info['ymin'])
            w = int(info['xmax']) - x1 +1
            h = int(info['ymax']) - y1 +1

            annotation = OrderedDict()
            annotation['segmentation'] = [[x1, y1, x1, (y1 + h), (x1 + w), (y1 + h), (x1 + w), y1]]
            annotation['area'] = w * h
            annotation['iscrowd'] = 0
            annotation['image_id'] = self.image_id
            annotation['bbox'] = [x1, y1, w, h]
            annotation['category_id'] = self.cat2id(str(info['name']))
            annotation['id'] = self.anno_id
            # annotation['ignore'] = int(obj['difficult'])
            self.annotations.append(annotation)

            self.anno_id+=1



    def append_categories(self, class_name):
        category = OrderedDict()
        category['supercategory'] = 'none'
        category['id'] = self.cat_id
        category['name'] = class_name
        self.categories.append(category)
        self.cat_id+=1

    def append(self, info):
        self.append_images(info['image'])
        self.append_annotations(info['annotations'])
        self.image_id+=1

    def run():
        pass


    def save(self, json_file='train.json'):
        ann = OrderedDict()
        ann['images'] = self.images
        ann['type'] = 'instances'
        ann['annotations'] = self.annotations
        ann['categories'] = self.categories

        with open(json_file, 'w') as f:
            json.dump(ann, f)

    @staticmethod
    def test():
        ann_file = './train.json'
        from pycocotools.coco import COCO
        coco = COCO(ann_file)
        annIds = coco.getAnnIds(imgIds=2893)
        print(coco.loadAnns(annIds))


