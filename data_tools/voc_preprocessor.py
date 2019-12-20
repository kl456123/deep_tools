# -*- coding: utf-8 -*-

import os
import numpy as np
import xml.etree.ElementTree as ET


from data_tools.core.preprocessor import Preprocessor


class VOCPreprocessor(Preprocessor):
    def read_labels(self, label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        labels_info = []
        for member in root.findall('object'):
            label_info = []
            for i in range(4):
                label_info.append(int(member[4][i].text))
            label_info.append(float(self.class2id[member[0].text]))
            labels_info.append(label_info)
        labels_info = np.asarray(labels_info).reshape(-1, 5)
        success = True if labels_info.shape[0] else False
        return labels_info, success


def main():
    # dir_names = []
    # dir_names = ['']
    # for dir_name in dir_names:
    input_dir = '/data/test_images'
    output_dir = '/data/test_images/test_h5'
    # input_dir = '/data/test_images/{}'.format(dir_name)
    # output_dir = '/data/tmp2/{}'.format(dir_name)
    input_size = (320, 320)
    preprocessor = VOCPreprocessor(
        input_dir,
        output_dir,
        input_size,
        ignore_error=True,
        use_crop=False)
    preprocessor.run()


if __name__ == '__main__':
    main()
