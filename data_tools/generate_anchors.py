from dataset_preprocess import get_coco
import numpy as  np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def calc_anchor(boxes):
    import ipdb
    ipdb.set_trace()
    kmeans = KMeans(n_clusters=2).fit(boxes[:, 2:])
    print(kmeans.cluster_centers_)
    plt.scatter(boxes[:, 2],boxes[:, 3], c=kmeans.labels_)
    plt.show()


def test():
    pass


def main(coco_jsons):
    boxes = []
    for coco_json in coco_jsons:
        coco = get_coco(coco_json)
        classes_name = ['excrement']
        ids = np.asarray(list((coco.imgs.keys())))
        for imgId in ids:
            annId = coco.getAnnIds(imgId)
            imgs_info = coco.loadImgs([imgId])[0]
            anns_info = coco.loadAnns(annId)
            for ann_info in anns_info:
                cat = coco.loadCats(ann_info['category_id'])[0]
                if cat['name'] in classes_name:
                    boxes.append(ann_info['bbox'])
        print(coco_json, len(boxes))

    boxes = np.asarray(boxes)
    calc_anchor(boxes)



if __name__=='__main__':
    main(['/data/tmp2/train2.json', '/data/tmp2/train.json'])
