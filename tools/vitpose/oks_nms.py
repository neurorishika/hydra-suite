"""OKS-NMS, transcribed verbatim from upstream mmpose/ViTPose.

Source: /tmp/vitpose-ref/nms.py (fetched from the ViTPose reference repo),
functions `oks_iou` and `oks_nms` -- lines 51-125 of that file. Copied
character-for-character (including the `list(...) and list(...)` construct in
`oks_iou`'s `vis_thr` branch, which is upstream's actual code, not a typo we
introduced) because Gate C requires bit-for-bit parity with the published
evaluation protocol, and hand-derived OKS-NMS numerics have been a repeat
source of silent regressions in this project.

Only import: numpy. No mmcv / mmpose dependency.
"""

from __future__ import annotations

import numpy as np


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = (
            np.array(
                [
                    0.26,
                    0.25,
                    0.25,
                    0.35,
                    0.35,
                    0.79,
                    0.79,
                    0.72,
                    0.72,
                    0.62,
                    0.62,
                    1.07,
                    1.07,
                    0.87,
                    0.87,
                    0.89,
                    0.89,
                ]
            )
            / 10.0
        )
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious


def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None, score_per_joint=False):
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k["score"].mean() for k in kpts_db])
    else:
        scores = np.array([k["score"] for k in kpts_db])

    kpts = np.array([k["keypoints"].flatten() for k in kpts_db])
    areas = np.array([k["area"] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(
            kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, vis_thr
        )

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep
