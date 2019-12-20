import cv2


def visualize_bbox(image, labels, size=(600, 800), keep_ratio=True):
    # if keep_ratio:
        # fx = size[0] / image.shape[1]
        # image = cv2.resize(image, (0, 0), fx=fx, fy=fx)
    # else:
        # image = cv2.resize(image, size)
    h, w = image.shape[:2]
    anns = labels['annotations']
    for ann in anns:
        text = ann['name']
        box = [
            ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']
        ]
        cv2.rectangle(
            image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            color=(255, 255, 255),
            thickness=2)
        cv2.putText(
            image,
            text, (int(box[0]), int(box[1])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0))
    title = 'detection'

    cv2.imshow(title, image)
    cv2.waitKey(0)

def visualize_bboxv2(image, labels, id2classes):
    h, w = image.shape[:2]
    for ann in labels:
        text = id2classes[str(int(ann[4]))]
        box = ann[:4]
        cv2.rectangle(
            image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            color=(255, 255, 255),
            thickness=2)
        cv2.putText(
            image,
            text, (int(box[0]), int(box[1])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0))
    title = 'detection'

    cv2.imshow(title, image)
    cv2.waitKey(0)
