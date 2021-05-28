import torch

from ..utils.decode import _nms, _topk, _transpose_and_gather_feat


def ctdet_decode(heat, wh, reg=None, K=100):
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
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections
