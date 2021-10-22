import math

import torch
import numpy as np

from CenterNet.utils.gaussian import draw_umich_gaussian, draw_msra_gaussian, gaussian_radius


class MultiPoseSample:
    def __init__(
        self, down_ratio=4, max_objects=128, gaussian_type="msra", num_joints=17
    ):

        self.down_ratio = down_ratio

        self.max_objects = max_objects
        self.gaussian_type = gaussian_type
        self.num_joints = num_joints

    @staticmethod
    def _coco_box_to_bbox(box):
        return np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32
        )

    def scale_point(self, point, output_size):
        x, y = point / self.down_ratio
        output_h, output_w = output_size

        x = np.clip(x, 0, output_w - 1)
        y = np.clip(y, 0, output_h - 1)

        return [x, y]

    def __call__(self, img, target):
        _, input_w, input_h = img.shape

        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio

        heatmap_keypoints = torch.zeros(
            (self.num_joints, output_h, output_w), dtype=torch.float32
        )
        keypoints = torch.zeros(
            (self.max_objects, self.num_joints * 2), dtype=torch.float32
        )
        keypoints_mask = torch.zeros(
            (self.max_objects, self.num_joints * 2), dtype=torch.bool
        )
        heatmap_keypoints_offset = torch.zeros(
            (self.max_objects * self.num_joints, 2), dtype=torch.float32
        )

        heatmap_keypoints_indices = torch.zeros(
            (self.max_objects * self.num_joints), dtype=torch.int64
        )
        heatmap_keypoints_mask = torch.zeros(
            (self.max_objects * self.num_joints), dtype=torch.bool
        )

        draw_gaussian = (
            draw_msra_gaussian if self.gaussian_type == "msra" else draw_umich_gaussian
        )

        num_objects = min(len(target), self.max_objects)
        for k in range(num_objects):
            ann = target[k]
            bbox = self._coco_box_to_bbox(ann["bbox"])

            # Scale to output size
            bbox[:2] = self.scale_point(bbox[:2], (output_h, output_w))
            bbox[2:] = self.scale_point(bbox[2:], (output_h, output_w))

            ct_int = torch.IntTensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                pts = torch.from_numpy(
                    np.array(ann["keypoints"], np.float32).reshape(self.num_joints, 3)
                )

                for j in range(self.num_joints):
                    if pts[j, 2] == 0:
                        continue

                    pts[j, :2] = torch.FloatTensor(
                        self.scale_point(pts[j, :2], (output_h, output_w))
                    )

                    keypoints[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                    keypoints_mask[k, j * 2: j * 2 + 2] = 1

                    pt_int = pts[j, :2].to(torch.int32)
                    heatmap_keypoints_offset[k * self.num_joints + j] = pts[j, :2] - pt_int
                    heatmap_keypoints_indices[k * self.num_joints + j] = (
                        pt_int[1] * output_w + pt_int[0]
                    )
                    heatmap_keypoints_mask[k * self.num_joints + j] = 1

                    draw_gaussian(heatmap_keypoints[j], pt_int, hp_radius)

        ret = {
            "heatmap_keypoints": heatmap_keypoints,
            "keypoints": keypoints,
            "keypoints_mask": keypoints_mask,
            "heatmap_keypoints_offset": heatmap_keypoints_offset,
            "heatmap_keypoints_indices": heatmap_keypoints_indices,
            "heatmap_keypoints_mask": heatmap_keypoints_mask,
        }

        return img, ret
