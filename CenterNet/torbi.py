
import os
import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

from CenterNet.datasets.coco import CocoDetection

from centernet_detection import CenterNetDetection
from transforms import ImageAugmentation, CategoryIdToClass
from transforms.ctdet import CenterDetectionSample
from transforms.multi_pose import MultiPoseSample
from transforms.sample import PoseFlip, MultiSampleTransform, ComposeSample


def test():
    transform = ComposeSample(
        [
            ImageAugmentation(
                iaa.Sequential(
                    [
                        iaa.Sequential(
                            [
                                iaa.Fliplr(0.5),
                                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                                iaa.LinearContrast((0.75, 1.5)),
                                iaa.AdditiveGaussianNoise(
                                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                                ),
                                iaa.Multiply((0.8, 1.2), per_channel=0.1),
                                iaa.Affine(
                                    scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
                                    translate_percent={
                                        "x": (-0.2, 0.2),
                                        "y": (-0.2, 0.2),
                                    },
                                    rotate=(-5, 5),
                                    shear=(-3, 3),
                                ),
                            ],
                            random_order=True,
                        ),
                        iaa.PadToFixedSize(width=512, height=512),
                        iaa.CropToFixedSize(width=512, height=512),
                    ]
                )
            )
        ]
    )

    dataset = CocoDetection(
        # os.path.join("/usr/home/tee/Developer/datasets/coco/images", "val2017"),
        os.path.join("/usr/home/tee/Developer/datasets/coco/images", "val2017"),
        # os.path.join("/usr/home/tee/Developer/datasets/coco/annotations", "instances_val2017.json"),
        os.path.join("/usr/home/tee/Developer/datasets/coco/annotations", "person_keypoints_val2017.json"),
        transforms=transform
    )

    # loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True
    # )

    items = enumerate(dataset)

    for _ in range(20):
        idx, (img, target) = next(items)

        # inp_img = np.moveaxis(img.squeeze().numpy(), 0, -1)
        # inp_img = ((inp_img + CocoDetection.mean) * CocoDetection.std)
        # BGR to RGB
        # inp_img = inp_img[:, :, ::-1]
        # hm = torch.sum(target['heatmap'].squeeze(), 0)
        # kps = torch.sum(target['heatmap_keypoints'].squeeze(), 0)
        # kps = target['heatmap_keypoints'].squeeze()[0]

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(cv2.resize(inp_img, hm.shape))
        # # plt.imshow(heatmap, 'jet', interpolation='none', alpha=0.2)
        # plt.imshow(kps, 'jet', interpolation='none', alpha=0.4)
        plt.show()


if __name__ == '__main__':
  test()
