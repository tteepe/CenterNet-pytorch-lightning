import torch
import torch.nn as nn


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _left_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = heat[i] >= heat[i - 1]
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _right_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = heat[i] >= heat[i + 1]
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _top_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = heat[i] >= heat[i - 1]
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _bottom_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = heat[i] >= heat[i + 1]
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _h_aggregate(heat, aggr_weight=0.1):
    return (
        aggr_weight * _left_aggregate(heat)
        + aggr_weight * _right_aggregate(heat)
        + heat
    )


def _v_aggregate(heat, aggr_weight=0.1):
    return (
        aggr_weight * _top_aggregate(heat)
        + aggr_weight * _bottom_aggregate(heat)
        + heat
    )


def sigmoid_clamped(x, clamp=1e-4):
    y = torch.clamp(x.sigmoid_(), min=clamp, max=1 - clamp)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = (
            tmp[:, e[1], ...].copy(),
            tmp[:, e[0], ...].copy(),
        )
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2, tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = (
            tmp[:, e[1], ...].copy(),
            tmp[:, e[0], ...].copy(),
        )
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)
