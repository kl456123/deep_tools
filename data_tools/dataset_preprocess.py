# -*- coding: utf-8 -*-
"""Convert PASCAL VOC annotations to MSCOCO format and save to a json file.
The MSCOCO annotation has following structure:
{
    "images": [
        {
            "file_name": ,
            "height": ,
            "width": ,
            "id":
        },
        ...
    ],
    "type": "instances",
    "annotations": [
        {
            "segmentation": [],
            "area": ,
            "iscrowd": ,
            "image_id": ,
            "bbox": [],
            "category_id": ,
            "id": ,
            "ignore":
        },
        ...
    ],
    "categories": [
        {
            "supercategory": ,
            "id": ,
            "name":
        },
        ...
    ]
}
"""


import json
from collections import OrderedDict
import sys
import os
import numpy as np
from random import shuffle
import glob
import h5py
from PIL import Image
import cv2



def main():
    dataset_list = ['coco_2017_val', 'coco_2017_train']
    classes = ['person']
    is_train = True
    train_transform = None
    target_transform = None
    datasets = build_dataset(dataset_list, transform=train_transform, target_transform=target_transform, is_train=is_train)[0]

    checked_id = get_checked_id(datasets, classes)

    converter = COCOConverter()

    for image, targets, index in datasets:
        info = get_info(image, targets, index)
        success = filter_info(info, checked_id)
        if success:
            converter.append(info)
        sys.stdout.write('\r {}/{}'.format(index, len(datasets)))
        # if index>300:
            # break

    converter.save()

def get_coco(ann_file):
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    return coco

def copy_images():
    import shutil
    src_dir =  '/data/COCO/trainval2017'
    dst_dir = '/home/indemind/Documents/SSD/datasets/trainval2017_person'
    coco = get_coco()

    ids = list(coco.imgs.keys())

    for image_id in ids:
        file_name = coco.loadImgs(image_id)[0]['file_name']
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.copy(src_path, dst_path)
        sys.stdout.write('\r {}/{}'.format(image_id, len(ids)))


def read_image(image_file):
    # image = Image.open(image_file).convert("RGB")
    # image = np.array(image)
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image





def test():
    ann_file = './train.json'
    coco = get_coco()
    print(coco.getCatIds(['1']))

def get_checked_id(concated_datasets, classes):
    datasets = concated_datasets.datasets[0]
    coco_ids = datasets.coco.getCatIds(classes)
    checked_id = []
    for coco_id in coco_ids:
        checked_id.append(datasets.coco_id_to_contiguous_id[coco_id])

    return checked_id

def get_info(image, targets, index):
    info = {}
    image_info = {'file_name': targets['file_name']}
    image_info['height'] = image.shape[0]
    image_info['width'] = image.shape[1]
    info['images'] = image_info

    boxes = targets['boxes']
    labels = targets['labels']
    annotations = []
    for ind, box in enumerate(boxes):
        anno_info = {}
        anno_info['xmin'] = box[0]
        anno_info['ymin'] = box[1]
        anno_info['xmax'] = box[2]
        anno_info['ymax'] = box[3]
        anno_info['name'] = labels[ind]
        annotations.append(anno_info)
    info['annotations'] = annotations

    return info

def filter_info(info, checked_id=[]):
    annotations = info['annotations']
    new_annotations = []
    for anno in annotations:
        if anno['name'] in checked_id:
            new_annotations.append(anno)
    if not new_annotations:
        return False

    # update before return
    info['annotations'] = new_annotations
    return True


class COCOConverter(object):
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.json_file = 'train.json'

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

            x1 = int(info['xmin']) - 1
            y1 = int(info['ymin']) - 1
            w = int(info['xmax']) - x1
            h = int(info['ymax']) - y1

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
        self.append_images(info['images'])
        self.append_annotations(info['annotations'])
        self.image_id+=1

    def run():
        pass
        # self.get_images()
        # self.get_annotations()
        # self.get_categories()


    def save(self, json_file='train.json'):
        ann = OrderedDict()
        ann['images'] = self.images
        ann['type'] = 'instances'
        ann['annotations'] = self.annotations
        ann['categories'] = self.categories

        with open(self.json_file, 'w') as f:
            json.dump(ann, f)


if __name__=='__main__':
    pass
    # main()
    # pass
    # test()
    # copy_images()
    # generate_anchors()
    # generate_hdf5()
