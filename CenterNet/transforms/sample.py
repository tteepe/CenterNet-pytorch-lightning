import torch
import torchvision
import numpy as np
from collections import Callable


class ComposeSample:
    """Composes several transforms together on sample of image and target

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class MultiSampleTransform:
    def __init__(self, transforms: [Callable]):
        self.transforms = transforms

    def __call__(self, img, target):
        ret_all = {}

        for transform in self.transforms:
            img, ret = transform(img, target)

            ret_all.update(ret)

        return img, ret_all


class PoseFlip:
    def __init__(self, flip_probability=0.5, num_joints=17):
        self.flip_probability = flip_probability

        self.num_joints = num_joints
        self.flip_idx_array = [
            0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
        ]

    def __call__(self, img, target):
        if torch.rand(1) < self.flip_probability:
            torchvision.transforms.RandomHorizontalFlip(img)

            for ann in target:
                if 'num_keypoints' not in ann or ann['num_keypoints'] == 0:
                    continue

                points = np.array(ann['keypoints'], np.float32).reshape(self.num_joints, 3)
                points_flipped = points[self.flip_idx_array, :]

                ann['keypoints'] = points_flipped.reshape(-1).tolist()

        return img, target
