# -*- coding: utf-8 -*-
import os
import numpy as np

from data_tools.core.preprocessor import Preprocessor
from data_tools.utils.util import get_coco


class COCOPreprocessor(Preprocessor):
    cocomap = {
        'person': 'person',
        'table': 'table',
        'dog': 'pet-dog',
        'cat': 'pet-cat',
        'key': 'key',
        'bed': 'bed'
    }

    def __init__(self, image_dir, label_path, output_dir, input_size):
        coco_json = label_path
        self.coco = get_coco(coco_json)

        super().__init__(image_dir, output_dir, input_size, single_label=True)

    @classmethod
    def filter_class(cls, classes):
        new_map = {}
        for class_name in classes:
            new_map[class_name] = cls.cocomap[class_name]
        cls.cocomap = new_map

    def get_paths_pair(self):
        images_path = []
        labels_path = []
        classes = [c for c in self.cocomap]
        catIds = self.coco.getCatIds(classes)
        imgIds = []
        for catId in catIds:
            imgIds.extend(self.coco.getImgIds(catIds=[catId]))

        if isinstance(self.root_dir, list):
            assert len(
                self.root_dir) == 1, 'only simple one input dir is supported'
            prefix = self.root_dir[0]
        else:
            prefix = self.root_dir

        for img in imgIds:
            images_path.append(
                os.path.join(prefix,
                             self.coco.loadImgs(img)[0]['file_name']))

        self.imgIds = imgIds
        return images_path, labels_path

    def read_labels(self, label_path):
        assert isinstance(label_path, int)
        image_id = label_path
        ann_ids = self.coco.getAnnIds(imgIds=self.imgIds[image_id])
        anns = self.coco.loadAnns(ann_ids)

        # filter some boxes
        labels_info = []
        for ann in anns:
            if ann['iscrowd']:
                continue
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
    classes = ['person']
    COCOPreprocessor.filter_class(classes)

    preprocessor = COCOPreprocessor(image_dir, label_path, output_dir,
                                    input_size)
    preprocessor.run()


if __name__ == '__main__':
    main()
