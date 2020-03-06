from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.generate_labels import generate_label
from pysot.utils.mot_tracker import MOTrackerWrapper

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--start_index', default=0, type=int,
                    help='start index of frame')
parser.add_argument('--class_name', default='wire', type=str,
                    help='class name')
args = parser.parse_args()


def get_classname(dirname):
    label_path = os.path.join(dirname, 'label.txt')
    if not os.path.exists(label_path):
        raise FileExistsError("no label file found")
    with open(label_path) as f:
        lines = f.readlines()
    return lines[0].strip()


def get_frames(video_name, start_index=0):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images.extend(glob(os.path.join(video_name, '*.png')))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images[start_index:]:
            frame = cv2.imread(img)
            yield frame, img


def check_rect(rect):
    for e in rect:
        if e != 0:
            return True
    return False


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    class_name = get_classname(args.video_name)
    mot_tracker = MOTrackerWrapper(tracker, class_name)

    for frame, image_path in get_frames(args.video_name, args.start_index):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if first_frame:
            init_rects = []
            while True:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                    # init_rect = (311, 136, 120, 125)
                except:
                    exit()
                if check_rect(init_rect):
                    init_rects.append(init_rect)
                else:
                    break
            mot_tracker.init(frame, init_rects)
            first_frame = False
        else:
            has_uncertainty = mot_tracker.track(frame)
            labels = mot_tracker.labels

            for bbox in labels:
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),
                                (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)
            if has_uncertainty or (cv2.waitKey(0)==ord('m')):
                first_frame = True
                continue
        labels = mot_tracker.labels
        if len(labels)==0:
            continue
        generate_label(labels, frame.shape[:2], path=image_path)


if __name__ == '__main__':
    main()
