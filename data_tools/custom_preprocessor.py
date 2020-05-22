# -*- coding: utf-8 -*-
import numpy as np
import json
import os

from data_tools.core.preprocessor import Preprocessor


class CustomPreprocessor(Preprocessor):
    def read_labels(self, label_path):
        with open(label_path, 'r') as f:
            label = json.load(f)

        # handle corner case
        if 'result' not in label:
            label = {'result': label}

        if len(label['result']['data']) == 0:
            return None, False
        ih = label['result']['data'][0]['ih']
        iw = label['result']['data'][0]['iw']

        labels_info = []
        for ann in label['result']['data']:
            xs = []
            ys = []
            cls = ann['category'][0]
            if cls not in self.class2id:
                continue
            for point in ann['points']:
                xs.append(point['x'])
                ys.append(point['y'])
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            xmin = xs.min()
            ymin = ys.min()
            xmax = xs.max()
            ymax = ys.max()
            labels_info.append([
                xmin * iw, ymin * ih, xmax * iw, ymax * ih,
                float(self.class2id[cls])
            ])
        return np.asarray(labels_info).reshape(-1, 5), True


def main():
    # dir_names = ['third_batch', 'fifth_batch',
                 # 'first_batch', 'second_batch', 'fourth_batch']
    # dir_names = ['first_batch']
    dir_names = ['12687', '7941']

    input_dirs = [os.path.join('/data/cleaner_machine/', dir_name)
                  for dir_name in dir_names]
    output_dir = '/data/test_images/cleaner_machine'
    input_size = (320, 320)
    classes = ['bg', 'person']
    CustomPreprocessor.filter_class(classes)
    preprocessor = CustomPreprocessor(
        input_dirs,
        output_dir,
        input_size,
        ignore_error=True,
        use_crop=False)

    preprocessor.run()


if __name__ == '__main__':
    main()
