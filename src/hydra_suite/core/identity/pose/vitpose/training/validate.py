from __future__ import annotations

import numpy as np
import torch

from ..config import HEATMAP_SIZE_WH
from ..decode import decode_udp_cv2
from ..transforms import transform_preds
from .loss import JointsMSELoss


def pck_from_preds(pred_xy, gt_xyv, bbox, thresholds) -> dict:
    gt_xyv = np.asarray(gt_xyv, np.float32)
    vis = gt_xyv[:, 2] > 0
    norm = float(np.sqrt(max(bbox[2] * bbox[3], 1e-6)))
    dist = np.linalg.norm(pred_xy[vis] - gt_xyv[vis, :2], axis=1) / norm
    out = {}
    for t in thresholds:
        out[t] = float((dist < t).mean()) if vis.any() else 0.0
    return out


def run_validation(model, loader, device, thresholds=(0.05, 0.1)) -> dict:
    model.eval()
    crit = JointsMSELoss(True)
    total_loss, n = 0.0, 0
    acc = {t: [] for t in thresholds}
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img)
            total_loss += (
                crit(out.cpu(), batch["target"], batch["target_weight"]).item()
                * img.shape[0]
            )
            n += img.shape[0]
            hm = out.cpu().numpy()
            coords, _ = decode_udp_cv2(hm, kernel=11)  # (B,K,2) heatmap space
            for b in range(img.shape[0]):
                pred = transform_preds(
                    coords[b],
                    batch["center"][b].numpy(),
                    batch["scale"][b].numpy(),
                    HEATMAP_SIZE_WH,
                )
                r = pck_from_preds(
                    pred,
                    batch["gt_joints"][b].numpy(),
                    batch["bbox"][b].numpy(),
                    thresholds,
                )
                for t in thresholds:
                    acc[t].append(r[t])
    return {
        "val_loss": total_loss / max(n, 1),
        "pck": {t: float(np.mean(v)) if v else 0.0 for t, v in acc.items()},
    }
