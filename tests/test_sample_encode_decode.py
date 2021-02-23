import json
import numpy as np
import torch
import torchvision
import imgaug.augmenters as iaa
import pytest

from decode.ctdet import ctdet_decode
from transforms import CategoryIdToClass, ComposeSample, ImageAugmentation
from transforms.ctdet import CenterDetectionSample


def test_cdet_encoding_decoding():
    sample_encoding = ComposeSample([
        ImageAugmentation(
            iaa.Identity(),  # change brightness, doesn't affect keypoints & bounding_boxes
            torchvision.transforms.ToTensor()
        ),
        CategoryIdToClass(range(0, 100)),
        CenterDetectionSample()
    ])

    img = (255 * np.random.rand(512, 512, 3)).astype(np.uint8)
    with open('tests/data/coco_annotation.json') as json_file:
        coco_annotation = json.load(json_file)

    ann_center = np.zeros((len(coco_annotation), 2))
    for i in range(len(coco_annotation)):
        x, y, w, h = coco_annotation[i]["bbox"]
        ann_center[i, 0] = x + w/2
        ann_center[i, 1] = y + h/2

    img, output = sample_encoding(img, coco_annotation)

    heatmap = output['hm'].unsqueeze(0)
    batch, cat, height, width = heatmap.size()
    wh = torch.zeros((batch, width, height, 2))
    reg = torch.zeros((batch, width, height, 2))

    indices = output['ind'].unsqueeze(0)
    indices_x = indices % width
    indices_y = indices // width
    wh[:, indices_y, indices_x] = output['wh'].unsqueeze(0)
    wh = wh.permute(0, 3, 1, 2)
    reg[:, indices_y, indices_x] = output['reg'].unsqueeze(0)
    reg = reg.permute(0, 3, 1, 2)
    detections = ctdet_decode(heatmap, wh, reg).squeeze().numpy()
    detections = 4 * detections[detections[:, 4] > 0.5]

    center = (detections[:, :2] + detections[:, 2:4]) / 2.

    assert abs(np.sum(center) - np.sum(ann_center)) == pytest.approx(0., abs=1e-3)


if __name__ == "__main__":
    test_cdet_encoding_decoding()
