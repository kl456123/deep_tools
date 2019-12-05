# -*- coding: utf-8 -*-
import os
import glob
import json
import pprint
import sys
import cv2
from visualize import visualize_bbox
from coco_converter import COCOConverter
import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert label file to coco format')
    parser.add_argument(
        '--output_path', type=str, help='output path to json file')
    parser.add_argument('--input_dir', type=str, help='input dir')
    parser.add_argument(
        '--input_size',
        nargs='+',
        type=int,
        help='input image size',
        default=600)

    args = parser.parse_args()
    return args


def generate_pairs(data_root,
                   label_suffix='.json',
                   image_suffix=['.jpg', '.JPG'],
                   filter_path=None):
    label_paths = []
    image_paths = []
    for json_dir in os.listdir(data_root):
        if filter_path is not None and json_dir not in filter_path:
            continue
        json_dir_path = os.path.join(data_root, json_dir)
        for scenario in os.listdir(json_dir_path):
            dir_path = os.path.join(json_dir_path, scenario)
            # get pairs here
            label_paths.extend(
                glob.glob(os.path.join(dir_path, "*{}".format(label_suffix))))
            for s in image_suffix:
                image_paths.extend(
                    glob.glob(os.path.join(dir_path, "*{}".format(s))))
    assert len(label_paths) == len(image_paths)
    return sorted(image_paths), sorted(label_paths)


def parse_label(label_file, image_file, size=None):
    dir_path = os.path.dirname(label_file)

    def get_info(label):
        # file_name = os.path.basename(image_file)
        file_name = image_file
        if len(label['result']['data']) == 0:
            return None, False
        ih = label['result']['data'][0]['ih']
        iw = label['result']['data'][0]['iw']
        if size:
            f = size[0] / iw
            iw = int(size[0])
            ih = int(f * ih)
        image_info = {'file_name': file_name, 'height': ih, 'width': iw}
        annotations_info = []

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
            annotations_info.append({
                'xmin': xmin * iw,
                'ymin': ymin * ih,
                'xmax': xmax * iw,
                'ymax': ymax * ih,
                'name': ann['category'][0]
            })
        return {'image': image_info, 'annotations': annotations_info}, True

    with open(label_file, 'r') as f:
        label = json.load(f)
        return get_info(label)


def check_label():
    data_root = "/data/cleaner_machine/first_batch"
    image_paths, label_paths = generate_pairs(
        data_root, filter_path=['json_12499_4091_20191017115501'])
    total_nums = len(image_paths)
    size = (600, 600)
    for ind, image_path in enumerate(image_paths):
        label, success = parse_label(label_paths[ind], image_path, size=size)
        if not success:
            continue
        pprint.pprint(label)
        image = cv2.imread(image_path)
        visualize_bbox(image, label, size)


def convert(data_root, output_path, size):
    converter = COCOConverter()
    data_root = data_root
    image_paths, label_paths = generate_pairs(data_root)
    total_nums = len(image_paths)
    failed_nums = 0
    for ind, image_path in enumerate(image_paths):
        label, success = parse_label(label_paths[ind], image_path, size=size)
        if not success:
            failed_nums += 1
            continue
        converter.append(label)
        sys.stdout.write('\r {}/{}'.format(ind+1, total_nums))
    converter.save(json_file=output_path)
    sys.stdout.write('\nfailed nums: {}\n'.format(failed_nums))


def main():
    args = parse_args()
    input_size = args.input_size
    input_dir = args.input_dir
    if len(input_size) == 1:
        input_size = (input_size, input_size)
    assert len(input_size) == 2
    assert os.path.exists(input_dir)
    convert(args.input_dir, args.output_path, args.input_size)


if __name__ == '__main__':
    # convert()
    main()
    # check_label()
