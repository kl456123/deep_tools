import torch
import glob
import h5py
import cv2

from data_tools.core.hdf5_converter import HDF5Converter
from data_tools.core.preprocessor import Preprocessor
from data_tools.utils.visualize import visualize_bboxv2


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_dir):
        super().__init__()
        self.h5_dir = h5_dir

        self.images, self.images_info, self.labels_info, classes = HDF5Converter.load(
            h5_dir)
        self.classes = classes
        self.id2classes = Preprocessor.generate_id2classes_map(self.classes)

    def __getitem__(self, index):
        # load
        image = self.images[index]
        image_info = self.images_info[index]
        label_info = self.labels_info[index]

        # reshape
        im_shape = (image_info[0], image_info[1], 3)
        image = image.reshape(im_shape)
        label_info = label_info.reshape(-1, 5)
        # convert to 3 channels gray image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image, image_info, label_info

    def __len__(self):
        return self.images.shape[0]

    def vis_from_sample(self, sample):
        image, image_info, label_info = sample
        visualize_bboxv2(image, label_info, self.id2classes)


def main():
    h5_dir = '/data/test_images/test_h5_no_crop'
    # h5_dir = '/data/test_images/cleaner_machine'
    dataset = HDF5Dataset(h5_dir)
    for ind, sample in enumerate(dataset):
        print(ind)
        dataset.vis_from_sample(sample)


if __name__ == '__main__':
    main()
