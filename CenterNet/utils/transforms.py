import numpy as np


def coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
