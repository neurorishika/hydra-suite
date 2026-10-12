from __future__ import annotations

import numpy as np
import torch

from ..decode import decode_udp_cv2, flip_back
from ..geometry import DEFAULT_GEOMETRY, PoseGeometry
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


def normalized_errors(pred_xy, gt_xyv, bbox) -> np.ndarray:
    """Per visible keypoint: pixel error / sqrt(animal box area)."""
    gt_xyv = np.asarray(gt_xyv, np.float32)
    vis = gt_xyv[:, 2] > 0
    norm = float(np.sqrt(max(bbox[2] * bbox[3], 1e-6)))
    return np.linalg.norm(pred_xy[vis] - gt_xyv[vis, :2], axis=1) / norm


def predict_heatmaps(model, img, flip_pairs=None, amp: bool = False):
    """``(raw output, heatmaps, flipped-back heatmaps or None)`` for a batch.

    With ``flip_pairs`` (a list, possibly empty) the mirrored image is also run
    and its heatmaps are mirrored back with left/right channels swapped.
    """
    use_amp = amp and img.device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        out = model(img)
        flipped = model(torch.flip(img, dims=[3])) if flip_pairs is not None else None
    hm = out.float().cpu().numpy()
    hm_flip = (
        flip_back(flipped.float().cpu().numpy(), flip_pairs)
        if flipped is not None
        else None
    )
    return out.float(), hm, hm_flip


def _to_source(coords, batch, b, geom):
    return transform_preds(
        coords[b],
        batch["center"][b].numpy(),
        batch["scale"][b].numpy(),
        geom.heatmap_size_wh,
    )


def run_validation(
    model,
    loader,
    device,
    thresholds=(0.05, 0.1),
    geom: PoseGeometry = DEFAULT_GEOMETRY,
    flip_pairs=None,
    amp: bool = False,
) -> dict:
    """Loss, PCK at ``thresholds`` and mean normalised error (``nme``) of the
    plain forward pass, which is what inference runs.

    ``flip_pairs`` (not None) adds ``flip_pck`` / ``flip_nme``: the plain and
    mirrored predictions averaged in source pixels.  The mirrored image's
    coordinates come back in the pixel-centre convention while labels use
    pixel edges (training mirrors x -> w - x), so they are moved +1 source
    pixel before averaging; averaging heatmaps instead would leave a -0.5 px
    bias.
    """
    model.eval()
    crit = JointsMSELoss(True)
    total_loss, n = 0.0, 0
    acc = {t: [] for t in thresholds}
    flip_acc = {t: [] for t in thresholds}
    errors, flip_errors = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out, hm, hm_flip = predict_heatmaps(model, img, flip_pairs, amp)
            total_loss += (
                crit(out.cpu(), batch["target"], batch["target_weight"]).item()
                * img.shape[0]
            )
            n += img.shape[0]
            coords, _ = decode_udp_cv2(hm, kernel=11)  # (B,K,2) heatmap space
            fcoords = (
                decode_udp_cv2(hm_flip, kernel=11)[0] if hm_flip is not None else None
            )
            for b in range(img.shape[0]):
                pred = _to_source(coords, batch, b, geom)
                gt = batch["gt_joints"][b].numpy()
                box = batch["bbox"][b].numpy()
                r = pck_from_preds(pred, gt, box, thresholds)
                for t in thresholds:
                    acc[t].append(r[t])
                errors.append(normalized_errors(pred, gt, box))
                if fcoords is not None:
                    fpred = _to_source(fcoords, batch, b, geom)
                    fpred[:, 0] += 1.0
                    avg = 0.5 * (pred + fpred)
                    fr = pck_from_preds(avg, gt, box, thresholds)
                    for t in thresholds:
                        flip_acc[t].append(fr[t])
                    flip_errors.append(normalized_errors(avg, gt, box))

    def _mean(chunks):
        flat = np.concatenate(chunks) if chunks else np.zeros(0)
        return float(flat.mean()) if flat.size else float("nan")

    result = {
        "val_loss": total_loss / max(n, 1),
        "pck": {t: float(np.mean(v)) if v else 0.0 for t, v in acc.items()},
        "nme": _mean(errors),
    }
    if flip_pairs is not None:
        result["flip_pck"] = {
            t: float(np.mean(v)) if v else 0.0 for t, v in flip_acc.items()
        }
        result["flip_nme"] = _mean(flip_errors)
    return result
