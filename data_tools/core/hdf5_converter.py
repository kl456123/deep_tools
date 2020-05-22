# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os
import glob


class HDF5Converter(object):
    KEY_IMAGES = 'images'
    KEY_LABELS_INFO = 'labels_info'
    KEY_IMAGES_INFO = 'images_info'
    KEY_META_DATA = 'meta_data'

    # wh
    def __init__(self, h5_dir, size=(160, 160), max_num=5000):
        self.h5_dir = h5_dir

        self.reset_buffer()
        self.num_lbl_info_cols = 5
        self.num_img_info_cols = 2

        self.max_num_samples_per_file = max_num

        self._file_index = 0

        if not os.path.exists(self.h5_dir):
            os.makedirs(self.h5_dir)
        self._file_index = 0
        self.meta_data = ''

        self.stored_amount = 0
        #  else:
        #  files = sorted([file for file in os.listdir(self.h5_dir)])
        #  self._file_index = len(files)
        # find the

    @property
    def total(self):
        return self.stored_amount + len(self.images)

    def reset_buffer(self):
        self.images = []
        self.labels_info = []
        self.images_info = []

    def generate_h5_path(self):
        h5_path = os.path.join(self.h5_dir, '{:03d}.h5'.format(
            self._file_index))
        self._file_index += 1
        return h5_path

    @property
    def full(self):
        return len(self.images) == self.max_num_samples_per_file

    @property
    def empty(self):
        return len(self.images) == 0

    def append(self, image, labels):
        self.images.append(image.flatten())
        self.images_info.append(np.asarray(image.shape[:2]))
        self.labels_info.append(labels.flatten())

        if self.full:
            self.save()

    def save(self):
        h5_path = self.generate_h5_path()
        num_images = len(self.images)

        h5_db = h5py.File(h5_path, mode='w')
        labels_info_dt = h5py.special_dtype(vlen=np.float32)
        image_dt = h5py.special_dtype(vlen=np.uint8)
        h5_db.create_dataset(
            self.KEY_IMAGES, (num_images, ), dtype=image_dt)
        h5_db.create_dataset(
            self.KEY_LABELS_INFO, (num_images, ),
            dtype=labels_info_dt)
        h5_db.create_dataset(
            self.KEY_IMAGES_INFO, (num_images, self.num_img_info_cols),
            dtype=np.int32)

        h5_db[self.KEY_IMAGES][...] = self.images

        h5_db[self.KEY_LABELS_INFO][...] = np.asarray(self.labels_info)
        h5_db[self.KEY_IMAGES_INFO][...] = np.asarray(self.images_info)

        # add meta data
        # h5_db[self.KEY_META_DATA] = "adsga"
        dt = h5py.special_dtype(vlen=str)
        dset = h5_db.create_dataset(self.KEY_META_DATA, (1, ), dtype=dt)
        dset[0] = str(self.meta_data)

        h5_db.close()
        # clear buffer
        self.reset_buffer()

        # update stored amount
        self.stored_amount += num_images

    def close(self):
        if not self.empty:
            self.save()

    @classmethod
    def load(cls, h5_dir):
        files = sorted(glob.glob(os.path.join(h5_dir, '*.h5')))
        # merge alll h5 files
        images = []
        images_info = []
        labels_info = []
        h5_dbs = []
        for file in files:
            h5_db = h5py.File(file, mode='r')
            h5_dbs.append(h5_db)
            images.append(h5_db[cls.KEY_IMAGES])
            images_info.append(h5_db[cls.KEY_IMAGES_INFO])
            labels_info.append(h5_db[cls.KEY_LABELS_INFO])
        images = np.concatenate(images)
        images_info = np.concatenate(images_info)
        labels_info = np.concatenate(labels_info)
        class_names = eval(h5_dbs[0][cls.KEY_META_DATA][0])['classes']

        for h5_db in h5_dbs:
            h5_db.close()
        return images, images_info, labels_info, class_names
