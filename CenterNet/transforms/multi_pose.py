import math

import numpy as np
import cv2

from utils.image import get_affine_transform, draw_msra_gaussian, draw_umich_gaussian, gaussian_radius, affine_transform
from utils.transforms import get_border, coco_box_to_bbox


class MultiPoseTransform:
    def __init__(self, image_transforms=None, target_transforms=None,
                 input_resolution=(512, 512), keep_resolution=False, padding=31, down_ratio=4,
                 augmented=False, random_crop=True, scale=0.4, shift=0.1, flip_probability=0.5,
                 rotation_probability=0, rotation_factor=0,
                 num_classes=80, max_objects=32, num_joints=17, gaussian_type='msra'):
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

        self.input_resolution = input_resolution
        # self.keep_resolution = keep_resolution
        # self.padding = padding
        self.down_ratio = down_ratio
        self.output_resolution = (self.input_resolution[0] // self.down_ratio,
                                  self.input_resolution[0] // self.down_ratio)

        self.augmented = augmented
        self.random_crop = random_crop
        self.scale = scale
        self.shift = shift
        self.flip_probability = flip_probability
        self.rotation_probability = rotation_probability
        self.rotation_factor = rotation_factor

        self.num_classes = num_classes
        self.max_objects = max_objects
        self.num_joints = num_joints
        self.flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                         [11, 12], [13, 14], [15, 16]]
        self.gaussian_type = gaussian_type

    def __call__(self, img, target):
        if self.target_transforms:
            target = self.target_transforms(target)

        height, width = img.size
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        rotation = 0

        flipped = False
        if self.augmented:
            if self.random_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = get_border(128, width)
                h_border = get_border(128, height)
                c[0] = np.random.randint(low=w_border, high=width - w_border)
                c[1] = np.random.randint(low=h_border, high=height - h_border)
            else:
                sf = self.scale
                cf = self.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.rotation_probability:
                rf = self.rotation_factor
                rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            if np.random.random() < self.flip_probability:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, rotation, self.input_resolution)
        inp = cv2.warpAffine(img, trans_input, self.input_resolution, flags=cv2.INTER_LINEAR)

        if self.image_transforms:
            inp = self.image_transforms(inp)

        trans_output_rot = get_affine_transform(c, s, rotation, self.output_resolution)
        trans_output = get_affine_transform(c, s, 0, self.output_resolution)

        hm = np.zeros((self.num_classes, self.output_resolution[1], self.output_resolution[0]), dtype=np.float32)
        hm_hp = np.zeros((self.num_joints, self.output_resolution[1], self.output_resolution[0]), dtype=np.float32)
        # dense_kps = np.zeros((num_joints, 2, output_res, output_res), dtype=np.float32)
        # dense_kps_mask = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        kps = np.zeros((self.max_objects, self.num_joints * 2), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        ind = np.zeros(self.max_objects, dtype=np.int64)
        reg_mask = np.zeros(self.max_objects, dtype=np.uint8)
        kps_mask = np.zeros((self.max_objects, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objects * self.num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objects * self.num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objects * self.num_joints), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.gaussian_type == "msra" else draw_umich_gaussian

        gt_det = []
        num_objects = min(len(target), self.max_objects)
        for k in range(num_objects):
            ann = target[k]
            bbox = coco_box_to_bbox(ann['bbox'])
            cls_id = int(ann['category_id']) - 1
            pts = np.array(ann['keypoints'], np.float32).reshape(self.num_joints, 3)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_resolution[1] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_resolution[0] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0) or (rotation != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(1e-5, int(radius))
                # radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * self.output_resolution[1] + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                num_kpts = pts[:, 2].sum()
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[k] = 0

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                # hp_radius = self.opt.hm_gauss \
                #     if self.opt.mse_loss else max(0, int(hp_radius))
                for j in range(self.num_joints):
                    if pts[j, 2] > 0:
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if 0 <= pts[j, 0] < self.output_resolution[0] and 0 <= pts[j, 1] < self.output_resolution[1]:
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * self.num_joints + j] = pts[j, :2] - pt_int
                            hp_ind[k * self.num_joints + j] = pt_int[1] * self.output_resolution[1] + pt_int[0]
                            hp_mask[k * self.num_joints + j] = 1
                            # if self.opt.dense_hp:
                            #     # must be before draw center hm gaussian
                            #     draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                            #                    pts[j, :2] - ct_int, radius, is_offset=True)
                            #     draw_gaussian(dense_kps_mask[j], ct_int, radius)
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                              pts[:, :2].reshape(self.num_joints * 2).tolist() + [cls_id])

        if rotation != 0:
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            kps_mask *= 0

        ret = {'hm': hm, 'reg': reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hps': kps, 'hps_mask': kps_mask, 'hm_hp': hm_hp,
               'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask}
        # if self.opt.dense_hp:
        #     dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
        #     dense_kps_mask = dense_kps_mask.reshape(
        #         num_joints, 1, output_res, output_res)
        #     dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
        #     dense_kps_mask = dense_kps_mask.reshape(
        #         num_joints * 2, output_res, output_res)
        #     ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
        #     del ret['hps'], ret['hps_mask']
        # if self.opt.reg_offset:
        #     ret.update({'reg': reg})
        # if self.opt.hm_hp:
        #     ret.update({'hm_hp': hm_hp})
        # if self.opt.reg_hp_offset:
        #     ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        # if self.opt.debug > 0 or not self.split == 'train':
        #     gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        #         np.zeros((1, 40), dtype=np.float32)
        #     meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
        #     ret['meta'] = meta
        # return ret

        return inp, ret

