import torch
import numpy as np

from common import _nms, _topk, _topk_channel, _transpose_and_gather_feat


def multi_pose_decode(heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat(
        [
            xs - wh[..., 0:1] / 2,
            ys - wh[..., 1:2] / 2,
            xs + wh[..., 0:1] / 2,
            ys + wh[..., 1:2] / 2,
        ],
        dim=2,
    )
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = (
            kps.view(batch, K, num_joints, 2).permute(0, 2, 1, 3).contiguous()
        )  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = (
            torch.stack([hm_xs, hm_ys], dim=-1)
            .unsqueeze(2)
            .expand(batch, num_joints, K, K, 2)
        )
        dist = ((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2
        )
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (
            (hm_kps[..., 0:1] < l)
            + (hm_kps[..., 0:1] > r)
            + (hm_kps[..., 1:2] < t)
            + (hm_kps[..., 1:2] > b)
            + (hm_score < thresh)
            + (min_dist > (torch.max(b - t, r - l) * 0.3))
        )
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)

    return detections


def multi_pose_post_process(dets, c, s, h, w):
    # dets: batch x max_dets x 40
    # return list of 39 in image coord
    ret = []
    for i in range(dets.shape[0]):
        # bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
        bbox = dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h)
        # pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
        pts = dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h)
        top_preds = (
            np.concatenate(
                [bbox.reshape(-1, 4), dets[i, :, 4:5], pts.reshape(-1, 34)], axis=1
            )
            .astype(np.float32)
            .tolist()
        )
        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
    return ret
