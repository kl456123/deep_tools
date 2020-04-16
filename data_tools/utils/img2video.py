# -*- coding: utf-8 -*-
import os
import shutil
import glob


def rename_images(image_dir):
    image_suffix = ''
    for ind, file in enumerate(sorted(os.listdir(image_dir))):
        src_path = os.path.join(image_dir, file)
        if not os.path.isfile(src_path):
            continue

        if os.path.splitext(src_path)[-1] not in ['.jpg', '.png']:
            continue
        suffix = os.path.splitext(src_path)[1]
        image_suffix = suffix
        dst_path = os.path.join(image_dir, '{:06d}{}'.format(ind, suffix))
        shutil.move(src_path, dst_path)
    return image_suffix


def to_video_helper(image_dir, image_suffix='.jpg', video_name='out', video_suffix='.mp4'):
    video_name = '{}{}'.format(video_name, video_suffix)
    command = 'ffmpeg -start_number 0 -i {}/%06d{}\
        -vcodec libx264 {}'.format(image_dir, image_suffix,
                                   os.path.join(image_dir, video_name))
    os.system(command)

    # rename_images(img_dir)


def to_video(image_dir):
    """
    Args:
        image_dir: string, contains all images needed to convert video
    """

    # get suffix of image at the same time
    image_suffix = rename_images(image_dir)
    if len(image_suffix) == 0:
        # empty dir
        return

    # call ffmpeg to convert image
    to_video_helper(image_dir, image_suffix)


def glob_helper(root_dir):
    # all_things = glob.glob('**/*', recursive=True)
    # dirs = [file if os.path.isdir(file) for file in all_things]

    img_dirs = ['./tmp{}/left'.format(i) for i in range(1, 6)]
    img_dirs = []
    for i in range(1, 6):
        dir_path = os.path.join(root_dir, 'tmp{}'.format(i))
        for sub_dir in ['left', 'result', 'depth']:
            img_dirs.append(os.path.join(dir_path, sub_dir))

    return img_dirs


if __name__ == '__main__':
    root_dir = './'
    img_dirs = glob_helper(root_dir)
    for img_dir in img_dirs:
        to_video(img_dir)
