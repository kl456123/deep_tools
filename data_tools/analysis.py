from dataset_preprocess import get_coco
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def calc_anchor(xy, title=''):
    kmeans = KMeans(n_clusters=2).fit(xy)
    print(kmeans.cluster_centers_)
    plt.title(title)
    plt.scatter(xy[:, 0], xy[:, 1], c=kmeans.labels_)
    plt.show()


def test():
    pass


def main(coco_jsons, classes_name):
    boxes = []
    total_nums = 0
    for coco in coco_jsons:
        # coco = get_coco(coco_json)
        # classes_name = ['person']
        ids = np.asarray(list((coco.imgs.keys())))
        for imgId in ids:
            annId = coco.getAnnIds(imgId)
            imgs_info = coco.loadImgs([imgId])[0]
            anns_info = coco.loadAnns(annId)
            for ann_info in anns_info:
                cat = coco.loadCats(ann_info['category_id'])[0]
                if cat['name'] in classes_name:
                    boxes.append(ann_info['bbox'])
            total_nums += len(anns_info)
        # print(coco_json, len(boxes))

    boxes = np.asarray(boxes)
    # print(classes_name, boxes.shape[0]/total_nums)
    xy = boxes[:, :2] + 0.5 * boxes[:, 2:]
    wh = boxes[:, 2:]
    calc_anchor(xy, title=classes_name[0]+' xy')
    calc_anchor(wh, title=classes_name[0]+' wh')
    return boxes.shape[0]


def calc_weights(results):
    # import ipdb
    # ipdb.set_trace()
    samples_per_cls = np.asarray(results)

    totals = samples_per_cls.sum()
    beta = 0.9
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(results)
    print(weights)


if __name__ == '__main__':
    classes_name = [
        'person', 'pet-cat', 'pet-dog', 'sofa', 'table', 'bed', 'excrement',
        'wire', 'key'
    ]
    cocos = []
    for coco_json in ['/data/tmp/train2.json']:
        cocos.append(get_coco(coco_json))
    results = []
    for class_name in classes_name:
        num = main(cocos, [class_name])
        results.append(num)
    # calc_weights(results)
    print(results)
