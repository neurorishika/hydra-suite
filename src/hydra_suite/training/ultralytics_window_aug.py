"""Centred window scale / aspect-ratio jitter for Ultralytics pose training.

Ultralytics' ``scale`` / ``translate`` augmentations zoom isotropically and
shift the crop centre. For canonical single-animal crops we want the opposite:
keep the centre fixed and vary the *window* -- its size (``scale_jitter``) and
width:height ratio (``aspect_jitter``) -- exactly as
``hydra_suite.training.scale_crop_aug.ScaleCropAug`` does for classifiers.

Ultralytics cannot express an anisotropic window, so this patches
``BaseDataset.get_image_and_label``: on the TRAINING dataset only, the loaded
image is re-cropped about its centre (pad value 114, Ultralytics' letterbox
grey), resized so its long side is ``imgsz`` again, and the labels (boxes,
keypoints, segments, ``cls``) are transformed with it so they stay exact.
Keypoints pushed outside the window get visibility 0; instances whose box is
mostly cut away are dropped, and if nothing would survive the sample is left
un-jittered rather than teaching a false "no animal" label.

The settings travel by environment variable because the Ultralytics CLI rejects
unknown arguments.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np

from .scale_crop_aug import ScaleCropAug

logger = logging.getLogger(__name__)

SCALE_ENV = "HYDRA_WINDOW_SCALE_JITTER"
ASPECT_ENV = "HYDRA_WINDOW_ASPECT_JITTER"
SEED_ENV = "HYDRA_WINDOW_JITTER_SEED"
PAD_VALUE = 114
# An instance is dropped when less than this fraction of its box survives.
MIN_BOX_KEEP = 0.25
_PATCH_ATTR = "_hydra_window_jitter_original"


def recrop_label(
    label: dict[str, Any], aug: ScaleCropAug, imgsz: int
) -> dict[str, Any]:
    """Re-crop ``label['img']`` about its centre and transform ``instances``.

    Returns ``label`` (mutated) or the unchanged input when the jitter does not
    apply (no instances, unsupported keypoint layout, or nothing would survive).
    """
    import cv2

    img = label.get("img")
    inst = label.get("instances")
    if img is None or inst is None or len(inst) == 0:
        return label
    kpts = inst.keypoints
    if kpts is not None and (kpts.ndim != 3 or kpts.shape[-1] != 3):
        return label
    h, w = img.shape[:2]
    out_w, out_h = aug.sample_window(h, w)
    if (out_w, out_h) == (w, h):
        return label

    tx, ty = (out_w - w) / 2.0, (out_h - h) / 2.0
    new_inst = inst[np.ones(len(inst), dtype=bool)]  # independent copy
    ori_format = new_inst._bboxes.format
    was_normalized = new_inst.normalized
    new_inst.convert_bbox(format="xyxy")
    new_inst.denormalize(w, h)
    new_inst.add_padding(tx, ty)
    before = np.maximum(new_inst.bbox_areas, 1e-9)
    new_inst.clip(out_w, out_h)
    keep = (new_inst.bbox_areas / before) >= MIN_BOX_KEEP
    if not keep.any():
        return label
    new_inst = new_inst[keep]
    if was_normalized:
        new_inst.normalize(out_w, out_h)
    new_inst.convert_bbox(format=ori_format)

    r = float(imgsz) / max(out_w, out_h)
    fw = max(1, min(int(math.ceil(out_w * r)), int(imgsz)))
    fh = max(1, min(int(math.ceil(out_h * r)), int(imgsz)))
    m = np.array([[r, 0.0, r * tx], [0.0, r, r * ty]], dtype=np.float64)
    new_img = cv2.warpAffine(
        img,
        m,
        (fw, fh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(PAD_VALUE,) * 3,
    )
    if new_img.ndim == 2 and img.ndim == 3:
        new_img = new_img[..., None]
    label["img"] = np.ascontiguousarray(new_img)
    label["instances"] = new_inst
    if "cls" in label and len(label["cls"]) == len(keep):
        label["cls"] = label["cls"][keep]
    label["resized_shape"] = (fh, fw)
    return label


def install_window_jitter(
    scale_jitter: float, aspect_jitter: float, seed: int | None = 0
) -> bool:
    """Patch ``BaseDataset.get_image_and_label``; returns True when installed."""
    if float(scale_jitter) <= 0.0 and float(aspect_jitter) <= 0.0:
        return False
    from ultralytics.data.base import BaseDataset

    original = getattr(
        BaseDataset.get_image_and_label, _PATCH_ATTR, BaseDataset.get_image_and_label
    )
    aug = ScaleCropAug(scale_jitter, aspect_jitter, seed=seed)

    def jittered(self: Any, index: int) -> dict[str, Any]:
        label = original(self, index)
        if not getattr(self, "augment", False):
            return label
        return recrop_label(label, aug, int(self.imgsz))

    setattr(jittered, _PATCH_ATTR, original)
    BaseDataset.get_image_and_label = jittered
    return True


def install_window_jitter_from_env() -> bool:
    """Install from ``HYDRA_WINDOW_*`` environment variables (no-op when unset)."""

    def _f(name: str) -> float:
        try:
            return float(os.environ.get(name, "0") or 0.0)
        except ValueError:
            return 0.0

    seed_raw = os.environ.get(SEED_ENV, "")
    try:
        seed = int(seed_raw) if seed_raw.strip() else 0
    except ValueError:
        seed = 0
    installed = install_window_jitter(_f(SCALE_ENV), _f(ASPECT_ENV), seed)
    if installed:
        logger.info(
            "Window jitter on: scale=%.2f aspect=%.2f", _f(SCALE_ENV), _f(ASPECT_ENV)
        )
    return installed
