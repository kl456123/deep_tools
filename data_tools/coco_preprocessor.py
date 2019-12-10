# -*- coding: utf-8 -*-

from inputs import Preprocessor
from dataset_preprocess import get_coco
import os
import numpy as np


class COCOPreprocessor(Preprocessor):
    def __init__(self, image_dir, label_path, output_dir, input_size):
        coco_json = label_path
        self.coco = get_coco(coco_json)

        self.cocomap = {
            'person': 'person',
            'table': 'table',
            'dog': 'pet-dog',
            'cat': 'pet-cat',
            'key': 'key',
            'bed': 'bed'
        }
        super().__init__(image_dir, output_dir, input_size, single_label=True)

    def get_paths_pair(self):
        images_path = []
        labels_path = []
        classes = [c for c in self.cocomap]
        catIds = self.coco.getCatIds(classes)
        imgIds = []
        for catId in catIds:
            imgIds.extend(self.coco.getImgIds(catIds=[catId]))

        for img in imgIds:
            images_path.append(
                os.path.join(self.root_dir,
                             self.coco.loadImgs(img)[0]['file_name']))

        self.imgIds = imgIds
        return images_path, labels_path

        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann],
                         np.float32).reshape((-1, 4))
        labels = np.array([self.get_label(obj["category_id"]) for obj in ann],
                          np.int64).reshape((-1, ))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def read_labels(self, label_path):
        assert isinstance(label_path, int)
        image_id = label_path
        ann_ids = self.coco.getAnnIds(imgIds=self.imgIds[image_id])
        anns = self.coco.loadAnns(ann_ids)

        labels_info = []
        for ann in anns:
            start_x = ann['bbox'][0]
            start_y = ann['bbox'][1]
            w = ann['bbox'][2]
            h = ann['bbox'][3]
            coco_name = self.coco.loadCats(ann['category_id'])[0]['name']
            if coco_name not in self.cocomap:
                continue
            name = self.cocomap[coco_name]
            labels_info.append([
                start_x, start_y, start_x + w, start_y + h,
                float(self.class2id[name])
            ])

        if len(labels_info) == 0:
            return None, False
        labels_info = np.asarray(labels_info).reshape(-1, 5)

        return labels_info, True


def main():
    # coco_dirs = ['/data/COCO/val2017', '/data/COCO/train2017']
    # image_dir = '/data/COCO/val2017'
    # label_path = '/data/COCO/annotations/instances_val2017.json'
    image_dir = '/data/COCO/train2017'
    label_path = '/data/COCO/annotations/instances_train2017.json'
    output_dir = '/data/tmp_memory/test_h5_coco'
    input_size = (320, 320)

    preprocessor = COCOPreprocessor(image_dir, label_path, output_dir,
                                    input_size)
    preprocessor.run()


if __name__ == '__main__':
    main()
