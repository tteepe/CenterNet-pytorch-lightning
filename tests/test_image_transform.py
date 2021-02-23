import json
import numpy as np
import torch
import torchvision
import imgaug.augmenters as iaa

from transforms import ImageAugmentation


def test_image_augmentation():
    img_aug = ImageAugmentation(
        iaa.Identity(),  # change brightness, doesn't affect keypoints & bounding_boxes
        torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.275, 0.275, 0.275], inplace=True)
        ])
    )

    img = (255 * np.random.rand(512, 512, 3)).astype(np.uint8)
    with open('data/coco_annotation.json') as json_file:
        coco_annotation = json.load(json_file)

    img_aug, ann_aug = img_aug(img, coco_annotation)

    # Output images
    assert type(img_aug) is torch.Tensor

    # Annotation
    for i in range(len(coco_annotation)):
        assert ann_aug[i]['keypoints'] == coco_annotation[i]['keypoints']
        assert ann_aug[i]['bbox'] == coco_annotation[i]['bbox']


if __name__ == "__main__":
    test_image_augmentation()
