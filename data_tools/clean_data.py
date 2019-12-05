# -*- coding: utf-8 -*-
import json
import sys

from parse_label import generate_pairs

def clean():
    data_root = "/data/cleaner_machine/first_batch"
    image_paths, label_paths = generate_pairs(data_root, filter_path=['20191107-1601'])
    total_nums = len(image_paths)
    for ind, image_path in enumerate(image_paths):
        with open(label_paths[ind], 'r') as f:
            label = json.load(f)
            # if 'image_url' not in label['data']:
                # modify the category
            for ann in label['result']['data']:
                ann['category'] = ['pet-dog']
            new_label = label
            # new_label = {}
            # new_label['data'] = {'image_url':image_path}
            # new_label['result'] = label
            with open(label_paths[ind], 'w') as f:
                json.dump(new_label,f)
        sys.stdout.write('\r {}/{}'.format(ind, total_nums))


if __name__=='__main__':
    clean()
