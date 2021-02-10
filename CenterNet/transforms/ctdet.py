import math

import numpy as np
import cv2

from utils.image import get_affine_transform, draw_msra_gaussian, draw_umich_gaussian, gaussian_radius, affine_transform


class CtDetTransform:
    def __init__(self, image_transforms=None, target_transforms=None,
                 resolution=(512, 512), keep_resolution=False, padding=31, down_ratio=4,
                 augmented=False, random_crop=True, scale=0.4, shift=0.1, flip_probability=0.5,
                 num_classes=80, max_objects=128, gaussian_type='msra'):
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

        self.resolution = resolution
        self.keep_resolution = keep_resolution
        self.padding = padding
        self.down_ratio = down_ratio

        self.augmented = augmented
        self.random_crop = random_crop
        self.scale = scale
        self.shift = shift
        self.flip_probability = flip_probability

        self.num_classes = num_classes
        self.max_objects = max_objects
        self.gaussian_type = gaussian_type

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __call__(self, img, target):
        if self.target_transforms:
            target = self.target_transforms(target)

        num_objects = min(len(target), self.max_objects)

        height, width = img.size
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        c = np.array([width / 2., height / 2.], dtype=np.float32)

        if self.keep_resolution:
            input_h = (height | self.padding) + 1
            input_w = (width | self.padding) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(height, width) * 1.0
            input_h, input_w = self.resolution

        flipped = False
        if self.augmented:
            if self.random_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, width)
                h_border = self._get_border(128, height)
                c[0] = np.random.randint(low=w_border, high=width - w_border)
                c[1] = np.random.randint(low=h_border, high=height - h_border)
            else:
                sf = self.scale
                cf = self.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.flip_probability:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,  (input_w, input_h), flags=cv2.INTER_LINEAR)

        if self.image_transforms:
            inp = self.image_transforms(inp)

        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        # dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        ind = np.zeros(self.max_objects, dtype=np.int64)
        reg_mask = np.zeros(self.max_objects, dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objects, self.num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objects, self.num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.gaussian_type == "msra" else draw_umich_gaussian

        gt_det = []
        for k in range(num_objects):
            ann = target[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = ann['class_id']
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(1e-5, int(radius))
                # radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                # if self.opt.dense_wh:
                #     draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg}
        # if self.opt.dense_wh:
        #     hm_a = hm.max(axis=0, keepdims=True)
        #     dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
        #     ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
        #     del ret['wh']
        # elif self.opt.cat_spec_wh:
        #     ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
        #     del ret['wh']
        # if self.opt.debug > 0 or not self.split == 'train':
        #     gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        #         np.zeros((1, 6), dtype=np.float32)
        #     meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
        #     ret['meta'] = meta

        return inp, ret

