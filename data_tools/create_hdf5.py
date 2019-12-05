# -*- coding: utf-8 -*-

from random import shuffle
import glob
import h5py
from PIL import Image
import sys
import numpy as np
import os
import cv2

from dataset_preprocess import get_coco, read_image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert images list to hdf5 file')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument(
        '--input_size', nargs='+', type=int, help='input image size', default=600)
    parser.add_argument(
        '--num_images_per_file',
        type=int,
        help='num images per file',
        default=300)
    parser.add_argument('--label_json', type=str, help='used for open coco json')

    args = parser.parse_args()
    return args

def get_hdf5_files(hdf5_dir):
    files = glob.glob(os.path.join(hdf5_dir, 'train_*.hdf5'))
    return files


def generate_hdf5(output_dir, num_images_per_file, json_file, size=None):
    hdf5_template = os.path.join(output_dir, 'train_{}.hdf5')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    coco = get_coco(json_file)
    ids = np.asarray(list((coco.imgs.keys())))

    num_totals = len(ids)
    num_files = np.ceil(num_totals / num_images_per_file).astype(np.int)

    def collect_batch(image_ids, file_id):
        images = []
        images_info = []
        for image_id in image_ids:
            image_fn = coco.loadImgs(int(image_id))[0]['file_name']
            # image_path = table[image_fn]
            image_path = image_fn
            image = read_image(image_path)
            if size is not None:
                fx = size[0] / image.shape[1]
                image = cv2.resize(image, (0, 0), fx=fx, fy=fx)
            # im2col
            images.append(image.flatten())
            images_info.append([image.shape[1], image.shape[0]])
            sys.stdout.write('\r{}/{}/{}/{}'.format(file_id, image_id,
                                                    len(image_ids),
                                                    num_totals))
        return np.asarray(images), np.asarray(images_info)

    sys.stdout.write('start create hdf5 files..\n')

    for i in range(num_files):

        start = i * num_images_per_file
        end = min(num_totals, start + num_images_per_file)

        images, images_info = collect_batch(ids[start:end], i)

        # create dataset
        hdf5_file = h5py.File(hdf5_template.format(i), mode='w')
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        hdf5_file.create_dataset("train_images", (end - start, ), dtype=dt)
        hdf5_file.create_dataset(
            'train_images_info', (end - start, 2), dtype=np.int32)

        hdf5_file['train_images'][...] = images
        hdf5_file['train_images_info'][...] = images_info
        hdf5_file.close()

def merge_hdf5(hdf5_dir, json_files):
    files = get_hdf5_files(hdf5_dir)
    h5_path = os.path.join(hdf5_dir, 'train_images.json')

    hdf5_file = h5py.File(h5_path, mode='w')
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    hdf5_file.create_dataset("train_images", (end - start, ), dtype=dt)
    hdf5_file.create_dataset(
        'train_images_info', (end - start, 2), dtype=np.int32)

    for file in files:
        pass

    hdf5_file['train_images'][...] = images
    hdf5_file['train_images_info'][...] = images_info
    hdf5_file.close()



def main():
    args = parse_args()
    output_dir = args.output_dir
    input_size = args.input_size
    if len(input_size) == 1:
        input_size = (input_size[0], input_size[0])
    assert len(input_size) == 2
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    assert os.path.exists(args.label_json)

    generate_hdf5(output_dir, args.num_images_per_file, args.label_json, input_size)


if __name__ == '__main__':
    # generate_hdf5((600, 600))
    main()
    # test()
    # read_image('./demo/000000000360.jpg')
