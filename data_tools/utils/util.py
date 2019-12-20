# -*- coding: utf-8 -*-
import os
import glob as glob_lib


def get_coco(ann_file):
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    return coco


def set_breakpoint():
    import ipdb
    ipdb.set_trace()


def glob(root_dir, suffix=[]):
    files = []
    for s in suffix:
        files.extend(
            glob_lib.glob(
                os.path.join(root_dir, '**/*{}'.format(s)),
                recursive=True))
    return files
