# -*- coding: utf-8 -*-

import cv2
import sys
import os
import random
import numpy as np
from data_tools.utils.util import glob
from auto_anno.tracking.generate_labels import generate_label


def check_light(pixel):
    light = np.zeros_like(pixel)
    light[...] = 230
    return (pixel >= light).all()


def check_valid_loc(loc, hw):
    return loc[0] >= 0 and loc[1] >= 0 and loc[0] < hw[1] and loc[1] < hw[0]


def bfs(obj_img):
    # start location
    mask = np.ones_like(obj_img[:, :, 0])
    hw = obj_img.shape[:2]
    roots = [(0, 0), (0, hw[0]-1), (hw[1]-1, 0), (hw[1]-1, hw[0]-1)]
    queue = []
    front_index = 0
    dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]
    visited = np.zeros_like(mask)
    hw = mask.shape[:2]

    for root in roots:
        queue.append(root)
        visited[root] = 1
        mask[root] = 0

    # extend all the pixel in the queue
    while front_index < len(queue):
        pixel_loc = queue[front_index]
        front_index += 1

        for dir_ in dirs:
            new_pixel_loc = pixel_loc[0]+dir_[0], pixel_loc[1]+dir_[1]
            # out of boundary
            if not check_valid_loc(new_pixel_loc, hw):
                continue

            # skip visited loc
            if visited[new_pixel_loc]:
                continue
            visited[new_pixel_loc] = 1

            # expand when is useful
            if check_light(obj_img[new_pixel_loc]):
                queue.append(new_pixel_loc)
                # mask out all pixel in queue
                mask[new_pixel_loc] = 0

    return mask.astype(np.bool)


def gen_obj_mask(obj_img):
    mask = bfs(obj_img)
    return mask


def gen_obj_box(mask):
    hw = mask.shape
    ys = np.arange(hw[0])
    xs = np.arange(hw[1])
    xs, ys = np.meshgrid(xs, ys)
    #  xs, ys = np.where(mask > -np.inf)
    xys = np.stack([ys, xs], axis=-1).reshape((hw[0], hw[1], 2))
    masked_pixels = xys[mask]
    x2 = masked_pixels[:, 1].max()
    x1 = masked_pixels[:, 1].min()
    y2 = masked_pixels[:, 0].max()
    y1 = masked_pixels[:, 0].min()

    return [x1, y1, x2, y2]


def crop(obj_img, box):
    obj_img = obj_img[box[1]:box[3], box[0]:box[2]]
    return obj_img


def handle_single(bg_image_path, obj_image_path, class_name='shoes'):
    bg_image = cv2.imread(bg_image_path)
    obj_image = cv2.imread(obj_image_path)
    #  cond = np.logical_and(obj_image[:, :, 0] == 255, obj_image[:, :, 1] == 255)
    #  cond = np.logical_and(cond, obj_image[:, :, 2] == 255)
    #  obj_image[cond] = 0
    #  cv2.imshow('test_orig_obj', obj_image)

    obj_mask = gen_obj_mask(obj_image)
    obj_box = gen_obj_box(obj_mask)

    #  cv2.imshow('test_mask', 0.5*obj_mask.astype(np.float32).astype(np.uint8))
    #  cv2.waitKey(0)

    # shrik the image
    obj_image = crop(obj_image, obj_box)
    obj_mask = crop(obj_mask, obj_box)
    #  cv2.imshow('test_mask', 0.5*obj_mask.astype(np.float32).astype(np.uint8))

    # h, w, c
    obj_img_size = obj_image.shape[:2]
    bg_img_size = bg_image.shape[:2]

    y = random.randint(0, bg_img_size[0]-obj_img_size[0])
    x = random.randint(0, bg_img_size[1]-obj_img_size[1])

    # first paste bg to obj_image
    obj_image[~obj_mask] = bg_image[y:obj_img_size[0] +
                                    y, x:obj_img_size[1]+x][~obj_mask]
    # paste
    bg_image[y:obj_img_size[0]+y,
             x:obj_img_size[1]+x] = obj_image

    bg_box = [x, y, obj_img_size[1]+x, obj_img_size[0]+y]

    #  cv2.imshow('test_res', bg_image)
    #  cv2.imshow('test_obj', obj_image)
    # cv2.imshow('test_mask', 0.5*obj_mask.astype(np.float32).astype(np.uint8))
    #  cv2.waitKey(0)
    bg_box.append(class_name)
    labels = [bg_box]
    return bg_image, labels


def test():
    bg_image_path = '/home/indemind/1000000000350960000.png'
    obj_image_path = '/data/shoes/ut-zap50k-images-square/Sandals/Flat/Gentle Souls/8123597.3.jpg'
    image, labels = handle_single(bg_image_path, obj_image_path)


def main():
    bg_image_path = '/home/indemind/1000000000350960000.png'
    obj_image_dir = '/data/shoes/ut-zap50k-images-square'
    data_dir = '/data/test_images/tmp79'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        print('directory is already exist')
    image_suffix = ['.jpg']
    obj_image_paths = glob(obj_image_dir, image_suffix)
    for ind, obj_image_path in enumerate(obj_image_paths):
        image, labels = handle_single(bg_image_path, obj_image_path)
        # save them
        saved_img_path = os.path.join(data_dir, '{:06d}.jpg'.format(ind))
        saved_label_path = os.path.join(
            '{}.xml'.format(os.path.splitext(saved_img_path)[0]))
        cv2.imwrite(saved_img_path, image)
        generate_label(labels, image.shape[:2], saved_label_path)
        sys.stdout.write('\r{}/{}'.format(ind, len(obj_image_paths)))
        sys.stdout.flush()


if __name__ == '__main__':
    #  test()
    main()
