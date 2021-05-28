import json
import numpy as np
import torch
import imgaug.augmenters as iaa

from CenterNet.transforms import ImageAugmentation, PoseFlip


def test_image_augmentation():
    img_aug_none = ImageAugmentation(
        iaa.Identity()
    )
    img_aug_change = ImageAugmentation(
        iaa.Fliplr(1)
    )

    sample_img = (255 * np.random.rand(512, 512, 3)).astype(np.uint8)
    with open('tests/data/coco_annotation.json') as json_file:
        coco_annotation = json.load(json_file)

    # Expect no change
    img_aug, ann_aug = img_aug_none(sample_img, coco_annotation)

    assert np.sum(img_aug[:, :, ::-1] - sample_img) == 0

    for i in range(len(coco_annotation)):
        assert ann_aug[i]['keypoints'] == coco_annotation[i]['keypoints']
        np.testing.assert_array_almost_equal(
            np.array(ann_aug[i]['bbox']),
            np.array(coco_annotation[i]['bbox'])
        )

    # Expect change
    img_aug, ann_aug = img_aug_change(sample_img, coco_annotation)

    assert np.sum(img_aug[:, :, ::-1] - sample_img) != 0

    for i in range(len(coco_annotation)):
        if ann_aug[i]["num_keypoints"] != 0:
            assert ann_aug[i]['keypoints'] != coco_annotation[i]['keypoints']
        assert ann_aug[i]['bbox'] != coco_annotation[i]['bbox']


def test_pose_flip():
    sample_img = torch.rand((1, 3, 512, 512))
    with open('tests/data/coco_annotation.json') as json_file:
        coco_annotation = json.load(json_file)

    flip = PoseFlip(1)

    # Flip
    img, ann = flip(sample_img, coco_annotation)

    assert torch.sum(img - sample_img) != 0
    for i in range(len(coco_annotation)):
        assert ann[i]['bbox'] != coco_annotation[i]['bbox']
        if ann[i]["num_keypoints"] == 0:
            assert ann[i]['keypoints'] == coco_annotation[i]['keypoints']
        else:
            assert ann[i]['keypoints'] != coco_annotation[i]['keypoints']

    # Flip back to original
    img, ann = flip(img, ann)

    assert torch.sum(img - sample_img) == 0
    for i in range(len(coco_annotation)):
        assert ann[i]['keypoints'] == coco_annotation[i]['keypoints']
        np.testing.assert_array_almost_equal(
            np.array(ann[i]['bbox']),
            np.array(coco_annotation[i]['bbox'])
        )


if __name__ == "__main__":
    test_image_augmentation()
    test_pose_flip()
