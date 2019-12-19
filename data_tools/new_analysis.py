# -*- coding: utf-8 -*-

from hdf5_dataset import HDF5Dataset
import numpy as np
import matplotlib.pyplot as plt


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
            # if xy[:, 1] < 100:
                # self.dataset.vis_from_sample(sample)
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


def main():
    h5_dir = '/data/tmp2/third_batch'
    analyzer = Analyzer(h5_dir)
    analyzer.analysis(filter_names='key')


if __name__ == '__main__':
    main()
