import torch

from ddd_utils import *

from utils.gaussian import transform_preds
from utils import _nms, _topk, _transpose_and_gather_feat


def get_pred_depth(depth):
    return depth


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def ddd_decode(heat, rot, depth, dim, wh=None, reg=None, K=40):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    depth = _transpose_and_gather_feat(depth, inds)
    depth = depth.view(batch, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)

    if wh is not None:
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        detections = torch.cat([xs, ys, scores, rot, depth, dim, wh, clses], dim=2)
    else:
        detections = torch.cat([xs, ys, scores, rot, depth, dim, clses], dim=2)

    return detections


def ddd_post_process_2d(dets, c, s, opt):
    # dets: batch x max_dets x dim
    # return 1-based class det list
    ret = []
    include_wh = dets.shape[2] > 16
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h)
        )
        classes = dets[i, :, -1]
        for j in range(opt.num_classes):
            inds = classes == j
            top_preds[j + 1] = np.concatenate(
                [
                    dets[i, inds, :3].astype(np.float32),
                    get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
                    get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
                    dets[i, inds, 12:15].astype(np.float32),
                ],
                axis=1,
            )
            if include_wh:
                top_preds[j + 1] = np.concatenate(
                    [
                        top_preds[j + 1],
                        transform_preds(
                            dets[i, inds, 15:17],
                            c[i],
                            s[i],
                            (opt.output_w, opt.output_h),
                        ).astype(np.float32),
                    ],
                    axis=1,
                )
        ret.append(top_preds)
    return ret


def ddd_post_process_3d(dets, calibs):
    # dets: batch x max_dets x dim
    # return 1-based class det list
    ret = []
    for i in range(len(dets)):
        preds = {}
        for cls_ind in dets[i].keys():
            preds[cls_ind] = []
            for j in range(len(dets[i][cls_ind])):
                center = dets[i][cls_ind][j][:2]
                score = dets[i][cls_ind][j][2]
                alpha = dets[i][cls_ind][j][3]
                depth = dets[i][cls_ind][j][4]
                dimensions = dets[i][cls_ind][j][5:8]
                wh = dets[i][cls_ind][j][8:10]
                locations, rotation_y = ddd2locrot(
                    center, alpha, dimensions, depth, calibs[0]
                )
                bbox = [
                    center[0] - wh[0] / 2,
                    center[1] - wh[1] / 2,
                    center[0] + wh[0] / 2,
                    center[1] + wh[1] / 2,
                ]
                pred = (
                    [alpha]
                    + bbox
                    + dimensions.tolist()
                    + locations.tolist()
                    + [rotation_y, score]
                )
                preds[cls_ind].append(pred)
            preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
        ret.append(preds)
    return ret
