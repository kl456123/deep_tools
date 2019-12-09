import os
import cv2
from coco_converter import COCOConverter
from visualize import visualize_bbox
import sys


def get_coco(ann_file):
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    return coco


def extract_data_from_coco(coco_dir,
                           ann_file,
                           classes,
                           classes_map,
                           converter=None,
                           size=(320, 320)):
    coco = get_coco(ann_file)
    catIds = coco.getCatIds(classes)
    imgIds = []
    for catId in catIds:
        imgIds.extend(coco.getImgIds(catIds=[catId]))

    # import ipdb
    # ipdb.set_trace()
    if converter:
        coco_converter = converter
    else:
        coco_converter = COCOConverter()
    for ind, imgId in enumerate(imgIds):
        annId = coco.getAnnIds(imgId)
        imgs_info = coco.loadImgs(imgId)[0]
        anns_info = coco.loadAnns(annId)

        image_path = os.path.join(coco_dir, imgs_info['file_name'])
        width = imgs_info['width']
        height = imgs_info['height']
        s_x = size[0] / width
        s_y = size[1] /height
        new_annos_info = []
        for ann_info in anns_info:
            if ann_info['category_id'] in catIds:
                bbox = ann_info['bbox']
                new_ann_info = {
                    'xmin': bbox[0] * s_x,
                    'ymin': bbox[1] * s_y,
                    'xmax': (bbox[0] + bbox[2]) * s_x,
                    'ymax': (bbox[1] + bbox[3]) * s_y,
                    'name':
                    classes_map[coco.cats[ann_info['category_id']]['name']]
                }
                new_annos_info.append(new_ann_info)
        imgs_info['file_name'] = os.path.join(coco_dir, imgs_info['file_name'])
        info = {'image': imgs_info, 'annotations': new_annos_info}
        coco_converter.append(info)

        # image = cv2.imread(image_path)
        # visualize_bbox(image, info, size=(600,600))
        sys.stdout.write('\r {}/{}'.format(ind + 1, len(imgIds)))
    sys.stdout.write('\n')


def main():
    coco_dirs = ['/data/COCO/val2017', '/data/COCO/train2017']
    ann_files = [
        '/data/COCO/annotations/instances_val2017.json',
        '/data/COCO/annotations/instances_train2017.json'
    ]
    classes = ['person', 'table', 'dog', 'cat', 'key', 'bed']
    classes_map = {
        'person': 'person',
        'table': 'table',
        'dog': 'pet-dog',
        'cat': 'pet-cat',
        'key': 'key',
        'bed': 'bed'
    }
    coco_converter = COCOConverter()
    for ind in range(len(coco_dirs)):
        extract_data_from_coco(coco_dirs[ind], ann_files[ind], classes,
                               classes_map, coco_converter)

    coco_converter.save(json_file='train.json')


if __name__ == '__main__':
    main()
