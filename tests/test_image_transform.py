import numpy as np
import torch
import torchvision
import imgaug.augmenters as iaa

from transforms import ImageAugmentation

coco_example_annotation =[{
    'segmentation': [[428.19, 219.47, 430.94, 209.57, 427.09, 219.74]],
    'num_keypoints': 15,
    'area': 2913.1104,
    'iscrowd': 0,
    'keypoints': [427, 170, 1,
                  429, 169, 2,
                  0, 0, 0,
                  434, 168, 2,
                  0, 0, 0,
                  441, 177, 2,
                  446, 177, 2,
                  437, 200, 2,
                  430, 206, 2,
                  430, 220, 2,
                  420, 215, 2,
                  445, 226, 2,
                  452, 223, 2,
                  447, 260, 2,
                  454, 257, 2,
                  455, 290, 2,
                  459, 286, 2],
    'image_id': 139,
    'bbox': [412.8, 157.61, 53.05, 138.01],
    'category_id': 1,
    'id': 230831
}]


def test_image_augmentation():
    img_aug = ImageAugmentation(
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect keypoints & bounding_boxes
        torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.275, 0.275, 0.275], inplace=True)
        ])
    )

    img = (255 * np.random.rand(512, 512, 3)).astype(np.uint8)

    img_aug, ann_aug = img_aug(img, coco_example_annotation)

    # Output images
    assert type(img_aug) is torch.Tensor

    # Annotation
    ann_aug = ann_aug[0]
    assert ann_aug['keypoints'] == coco_example_annotation[0]['keypoints']
    assert ann_aug['bbox'] == coco_example_annotation[0]['bbox']


if __name__ == "__main__":
    test_image_augmentation()
