"""GATE C: reproduce published ViTPose COCO val AP.

Top-down AP is only comparable to published numbers when evaluated against the
STANDARD person detections (not ground-truth boxes, which score higher).

pycocotools is sufficient here: xtcocotools' default sigmas are allclose to
pycocotools' COCO sigmas and the full stats vector is identical. xtcocotools
also does not install on Python 3.13 from PyPI.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hydra_suite.core.identity.pose.vitpose.decode import decode_udp_torch, flip_back
from hydra_suite.core.identity.pose.vitpose.transforms import (
    box2cs,
    normalize,
    top_down_affine,
    transform_preds,
)
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
from hydra_suite.core.identity.pose.vitpose.weights import load_checkpoint
from tools.vitpose.oks_nms import oks_nms

ASSET_DIR = Path(os.path.expanduser("~/.cache/vitpose-assets"))
COCO_FLIP_PAIRS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]
DET_SCORE_THR = 0.0  # upstream keeps all detections and lets OKS sort it out

# Upstream ViTPose-B COCO val eval config (/tmp/vitpose-ref/vitpose_b_cfg.py
# lines 84-90): plain (non-soft) oks_nms, oks_thr=0.9, vis_thr=0.2.
OKS_THR = 0.9
VIS_THR = 0.2


def evaluate(
    variant: str,
    head: str,
    ckpt: Path,
    device: str = "cpu",
    limit: int | None = None,
    batch_size: int = 16,
) -> dict[str, float]:
    ann_file = ASSET_DIR / "annotations" / "person_keypoints_val2017.json"
    det_file = ASSET_DIR / "COCO_val2017_detections_AP_H_56_person.json"
    coco = COCO(str(ann_file))
    dets = json.loads(det_file.read_text())
    dets = [d for d in dets if d["category_id"] == 1 and d["score"] > DET_SCORE_THR]
    if limit is not None:
        keep = set(sorted({d["image_id"] for d in dets})[:limit])
        dets = [d for d in dets if d["image_id"] in keep]

    model = build_vitpose(variant, head).eval().to(device)
    load_checkpoint(model, ckpt, strict=True)

    results = []
    for start in range(0, len(dets), batch_size):
        chunk = dets[start : start + batch_size]
        crops, metas = [], []
        for d in chunk:
            img_path = (
                ASSET_DIR / "val2017" / coco.loadImgs(d["image_id"])[0]["file_name"]
            )
            img = cv2.imread(str(img_path))
            c, s = box2cs(np.array(d["bbox"], np.float32))
            crops.append(normalize(top_down_affine(img, c, s)))
            metas.append((d, c, s))
        batch = torch.from_numpy(np.stack(crops)).to(device)
        with torch.no_grad():
            hm = model(batch)
            # Flip test: every ViTPose config sets flip_test=True.
            hm_flip = model(torch.flip(batch, dims=[3]))
            hm_flip = torch.from_numpy(
                flip_back(hm_flip.cpu().numpy(), COCO_FLIP_PAIRS)
            ).to(device)
            # With UDP, shift_heatmap must stay False -- do NOT column-shift.
            hm = (hm + hm_flip) * 0.5
            coords, maxvals = decode_udp_torch(hm)
        coords_np = coords.cpu().numpy()
        vals_np = maxvals.cpu().numpy()
        for i, (d, c, s) in enumerate(metas):
            kpts = transform_preds(coords_np[i], c, s, (48, 64))
            # Upstream area (mmpose TopdownHeatmapBaseHead.decode):
            # all_boxes[:, 4] = np.prod(s * 200.0, axis=1), where `s` is the
            # same (pixel_std-normalized, padded) scale used for the affine
            # crop -- i.e. exactly the `s` returned by box2cs above.
            area = float(np.prod(s * 200.0))
            results.append(
                {
                    "image_id": d["image_id"],
                    "category_id": 1,
                    "keypoints": np.concatenate([kpts, vals_np[i]], axis=1),
                    "area": area,
                    "score": float(d["score"]),
                }
            )

    # Rescoring + OKS-NMS, transcribed from upstream
    # TopDownCocoDataset.evaluate (/tmp/vitpose-ref/topdown_coco_dataset.py
    # lines 296-320): per image, replace each detection's score with
    # mean(keypoint scores > vis_thr) * box_score (0 if none clear the
    # threshold), then run plain oks_nms per image to drop near-duplicate
    # poses. Without this, ~27 unsuppressed detections/image inflate false
    # positives and depress AP even though localization is correct.
    by_image: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_image[r["image_id"]].append(r)

    final_results = []
    for image_id, img_kpts in by_image.items():
        for n_p in img_kpts:
            kpt_scores = n_p["keypoints"][:, 2]
            valid = kpt_scores[kpt_scores > VIS_THR]
            kpt_score = float(valid.mean()) if len(valid) else 0.0
            n_p["score"] = kpt_score * n_p["score"]

        keep = oks_nms(img_kpts, OKS_THR, sigmas=None, vis_thr=None)
        for k in keep:
            n_p = img_kpts[k]
            final_results.append(
                {
                    "image_id": n_p["image_id"],
                    "category_id": n_p["category_id"],
                    "keypoints": n_p["keypoints"].reshape(-1).tolist(),
                    "score": n_p["score"],
                }
            )

    dt = coco.loadRes(final_results)
    e = COCOeval(coco, dt, "keypoints")
    # COCOeval defaults to evaluating every image_id in the ground truth
    # (5000 for val2017). Our results only cover the images that have
    # standard-detection boxes (3,893 for the full run; fewer under `limit`).
    # Leaving imgIds unset silently counts every uncovered image as pure
    # false negatives and craters AP -- restrict to exactly the images we
    # produced detections for, matching upstream top-down eval convention.
    e.params.imgIds = sorted({d["image_id"] for d in dets})
    e.evaluate()
    e.accumulate()
    e.summarize()
    return {"AP": float(e.stats[0])}
