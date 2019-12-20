# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from data_tools.core.hdf5_dataset import HDF5Dataset


class Analyzer():
    def __init__(self, h5_dir):
        # h5_dir = '/data/tmp2/second_batch'
        self.dataset = HDF5Dataset(h5_dir)
        self.classes = self.dataset.classes

    def condition(self):
        pass

    def analysis(self, filter_classes=None, filter_names=None):
        if filter_names:
            filter_classes = self.classes.index(filter_names)
        xys = []
        whs = []
        # import ipdb
        # ipdb.set_trace()
        for sample in self.dataset:
            image, image_info, label_info = sample
            # xyxy
            boxes = label_info[:, :4]
            classes = label_info[:, 4]
            if filter_classes:
                keep = classes == filter_classes
                boxes = boxes[keep]
            if boxes.shape[0] == 0:
                continue
            xy = (boxes[:, :2]+boxes[:, 2:])/2
            wh = boxes[:, 2:]-boxes[:, :2]
            xys.append(xy)
            whs.append(wh)
        xys = np.concatenate(xys, axis=0).reshape(-1, 2)
        xys[:, 1] = 320-xys[:, 1]
        whs = np.concatenate(whs, axis=0).reshape(-1, 2)

        self._analysis_data(xys, 'xys {}'.format(filter_names))
        self._analysis_data(whs, 'whs {}'.format(filter_names))

        self.calc_anchor(whs)

    def _analysis_data(self, data, title=''):
        plt.title(title)
        plt.scatter(data[:, 0], data[:, 1])
        plt.show()

    def calc_anchor(self, whs, k=3):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k).fit(whs)
        print(kmeans.cluster_centers_)
        plt.title('anchors')
        plt.scatter(whs[:, 0], whs[:, 1], c=kmeans.labels_)
        plt.show()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='convert images list to hdf5 file')
    parser.add_argument('--input_dir', type=str,
                        help='input dir of h5', default='/data/tmp2/fifth_batch')
    parser.add_argument('--class_name', type=str,
                        help='class name to analysis', default='excrement')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    h5_dir = args.input_dir
    class_name = args.class_name
    analyzer = Analyzer(h5_dir)
    analyzer.analysis(filter_names=class_name)


if __name__ == '__main__':
    main()
