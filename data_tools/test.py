

"""
read image from hdf5
read label from coco json
finally visualize it
"""
import argparse
import glob
import os
import h5py
from dataset_preprocess import get_coco
from visualize import visualize_bbox

def parse_args():
    parser = argparse.ArgumentParser(
        description='convert label file to coco format')
    parser.add_argument('--hdf5_dir', type=str, help='hdf5 dir where images are stored in')
    parser.add_argument('--json_file', type=str, help='coco json')

    args = parser.parse_args()
    return args




def main():
    args = parse_args()
    test(args.json_file, args.hdf5_dir)

def get_targets(anns):
    num = len(anns)
    targets = {}
    annotations = []
    for ind in range(num):
        x1, y1, w, h = anns[ind]['bbox']
        classname = 'sadga'
        annotations.append({
            'name': str(classname),
            'xmin': x1,
            'ymin': y1,
            'xmax': x1+w,
            'ymax': y1+h
        })
    targets['annotations'] = annotations
    targets['image'] = {}
    return targets


def test(json_file, hdf5_dir):

    # hdf5_path = '/data/cleaner_machine/train_hdf5/train_0.hdf5'
    files = sorted(glob.glob(os.path.join(hdf5_dir, 'train_*.hdf5')))
    assert len(files)>0
    hdf5_path = files[0]

    db = h5py.File(hdf5_path, 'r')
    images = db['train_images']
    images_info = db['train_images_info']
    coco = get_coco(json_file)
    num = len(images)

    for image_id in range(num):
        image_fn = coco.loadImgs(int(image_id))[0]['file_name']
        annIds = coco.getAnnIds(imgIds=int(image_id))
        anns = coco.loadAnns(annIds)
        shape = (images_info[image_id][1], images_info[image_id][0], 3)
        image = images[image_id].reshape(shape)
        image = image[:, :, ::-1]
        targets = get_targets(anns)
        visualize_bbox(image, targets)


if __name__=='__main__':
    main()
