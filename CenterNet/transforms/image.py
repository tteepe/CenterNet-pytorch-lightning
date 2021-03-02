import copy

import cv2
import numpy as np

from imgaug.augmentables import Keypoint, KeypointsOnImage, BoundingBox, BoundingBoxesOnImage
from imgaug.augmenters import Augmenter, Identity


class ImageAugmentation:
    def __init__(self, imgaug_augmenter: Augmenter = Identity(), img_transforms=None, num_joints=17):
        self.ia_sequence = imgaug_augmenter
        self.img_transforms = img_transforms
        self.num_joints = num_joints

    def __call__(self, img, target):
        # PIL to array BGR
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        target = copy.deepcopy(target)

        # Prepare augmentables for imgaug
        bounding_boxes = []
        keypoints = []
        for idx in range(len(target)):
            ann = target[idx]

            # Bounding Box
            box = ann['bbox']
            bounding_boxes.append(BoundingBox(
                x1=box[0],
                y1=box[1],
                x2=box[0] + box[2],
                y2=box[1] + box[3],
                label=idx
            ))

            # Keypoints
            if 'num_keypoints' not in ann or ann['num_keypoints'] == 0:
                continue

            points = np.array(ann['keypoints'], np.float32).reshape(self.num_joints, 3)
            for i in range(self.num_joints):
                keypoints.append(Keypoint(x=points[i][0], y=points[i][1]))

        # Augmentation
        image_aug, bbs_aug, kps_aug = self.ia_sequence(
            image=img,
            bounding_boxes=BoundingBoxesOnImage(bounding_boxes, shape=img.shape),
            keypoints=KeypointsOnImage(keypoints, shape=img.shape)
        )

        # Write augmentation back to annotations
        for bb in bbs_aug:
            target[bb.label]['bbox'] = [
                bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1
            ]

        for ann in target:
            if 'num_keypoints' not in ann or ann['num_keypoints'] == 0:
                continue

            aug_keypoints = []
            points = np.array(ann['keypoints'], np.float32).reshape(self.num_joints, 3)
            for i in range(self.num_joints):
                aug_kp = kps_aug.items.pop(0)
                kp_type = int(points[i][2])
                if kp_type == 0:
                    aug_keypoints.extend([0, 0, 0])
                else:
                    aug_keypoints.extend([aug_kp.x, aug_kp.y, kp_type])

            ann['keypoints'] = aug_keypoints

        # torchvision transforms
        if self.img_transforms:
            image_aug = self.img_transforms(image_aug)

        return image_aug, target
