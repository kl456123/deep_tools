# -*- coding: utf-8 -*-

import cv2

from data_tools.utils.visualize import visualize_bboxv2


def main():
    image_path = '/data/cleaner_machine/first_batch/json_12480_4077_20191017115301/bathroom/IMG20190921152801.jpg'
    image = cv2.imread(image_path)
    label_info = [[1804.1932800000002, 0.0, 2604.754944, 2804.944896, 1.0]]
    visualize_bboxv2(image, label_info, {'0':'bg', '1':'person'})


if __name__ == '__main__':
    main()
