from dict2xml import dict2xml
from collections import OrderedDict
import os
# <annotation>
#   <folder>images</folder>
#   <filename>000182.jpg</filename>
#   <path>C:\Users\breakpoint\Documents\WeChat Files\xlconfidence\FileStorage\File\2020-02\images\images\000182.jpg</path>
#   <source>
#     <database>Unknown</database>
#   </source>
#   <size>
#     <width>640</width>
#     <height>400</height>
#     <depth>1</depth>
#   </size>
#   <segmented>0</segmented>
#   <object>
#     <name>wire</name>
#     <pose>Unspecified</pose>
#     <truncated>0</truncated>
#     <difficult>0</difficult>
#     <bndbox>
#       <xmin>491</xmin>
#       <ymin>252</ymin>
#       <xmax>616</xmax>
#       <ymax>282</ymax>
#     </bndbox>
#   </object>
# </annotation>


def generate_label(labels, im_shape, path):
    filename = os.path.basename(path)
    dirname = os.path.dirname(path)
    saved_fn = os.path.join(dirname, '{}.xml'.format(
        os.path.splitext(filename)[0]))
    objects = []
    for label in labels:
        single_object = OrderedDict()
        single_object.update({'name': label[4], 'pose': 'Unspecified', 'truncated': 0, 'difficult': 0, 'bndbox': {
            'xmin': label[0], 'ymin': label[1], 'xmax': label[2], 'ymax': label[3]}})
        objects.append(single_object)

    label_dict = OrderedDict()
    label_dict.update({'annotation': {'folder': 'images', 'filename': filename, 'path': path, 'source': {
        'database': 'Unknown'}, 'size': {'width': im_shape[1], 'height': im_shape[0], 'depth': 1}, 'segmented': 0, 'object': objects}})
    xml = dict2xml(label_dict)
    with open(saved_fn, 'w') as f:
        f.write(xml)


def main():
    filename = '000182.jpg'
    labels = [[491, 252, 616, 282]]
    generate_label(labels, path='')


if __name__ == "__main__":
    main()
