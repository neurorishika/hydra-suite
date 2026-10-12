from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..geometry import DEFAULT_GEOMETRY, PoseGeometry
from ..transforms import affine_matrix, box2cs, normalize, top_down_affine
from .augment import AugmentParams, augment_pixels, sample_rotation
from .targets import generate_udp_gaussian
from .window_jitter import WindowJitter

# Kept for backward compatibility: some callers/tests reach for the
# process-wide default stride directly. Per-instance geometry now lives on
# CocoKeypointsDataset._feat_stride (see __init__). This constant's value is
# bit-identical to its pre-branch value ([4.06383, 4.047619] for
# DEFAULT_GEOMETRY's 192x256 / 48x64), since DEFAULT_GEOMETRY reproduces the
# historical 192x256 constants exactly.
FEAT_STRIDE = (np.array(DEFAULT_GEOMETRY.image_size_wh, np.float32) - 1.0) / (
    np.array(DEFAULT_GEOMETRY.heatmap_size_wh, np.float32) - 1.0
)

# legacy random zoom (scale_jitter=None); rotation and photometric ranges
# live in augment.AugmentParams
_SCALE_JITTER = 0.30


def load_coco_index(dataset_dir: Path) -> tuple[list[int], dict]:
    coco = json.loads(
        (Path(dataset_dir) / "annotations.json").read_text(encoding="utf-8")
    )
    images = {img["id"]: img for img in coco["images"]}
    anns = [a for a in coco["annotations"] if a.get("num_keypoints", 0) > 0]
    index = {a["id"]: (a, images[a["image_id"]]) for a in anns}
    return list(index.keys()), index


def image_path(dataset_dir: Path, file_name: str) -> Path:
    """An annotation's image file.

    PoseKit's COCO export names files relative to the dataset folder
    (``images/x.png``, standard COCO); bare names (``x.png``) live in
    ``images/``.
    """
    rel = Path(file_name)
    if len(rel.parts) > 1:
        return Path(dataset_dir) / rel
    return Path(dataset_dir) / "images" / rel


def _warp_joints(joints_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return joints_xy @ matrix[:, :2].T + matrix[:, 2]


class CocoKeypointsDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        ids: list[int],
        sigma: float,
        augment: bool,
        geom: PoseGeometry = DEFAULT_GEOMETRY,
        scale_jitter: float | None = None,
        aspect_jitter: float = 0.0,
        seed: int = 0,
        flip_pairs: list | None = None,
        fliplr: float = 0.0,
        flipud: float = 0.0,
        aug: AugmentParams | None = None,
        ignore_occluded: bool = False,
    ) -> None:
        # ``scale_jitter=None`` keeps the legacy hard-coded random zoom. Any
        # explicit value (including 0) switches to the centred window jitter:
        # the crop window's size (``scale_jitter``) and width:height ratio
        # (``aspect_jitter``) vary about the fixed crop centre.
        self._window = (
            WindowJitter(scale_jitter or 0.0, aspect_jitter, seed)
            if augment and (scale_jitter is not None or aspect_jitter > 0.0)
            else None
        )
        self._legacy_scale = scale_jitter is None
        self.dir = Path(dataset_dir)
        self.ids, self.index = load_coco_index(dataset_dir)
        self.ids = [i for i in ids if i in self.index]
        self.sigma = float(sigma)
        self.augment = bool(augment)
        self.geom = geom
        num_kpts = len(self.index[self.ids[0]][0]["keypoints"]) // 3 if self.ids else 0
        self._flip_idx = np.asarray(flip_index(flip_pairs or [], num_kpts), np.int64)
        self._fliplr = float(fliplr) if augment else 0.0
        self._flipud = float(flipud) if augment else 0.0
        self.aug = aug if aug is not None else AugmentParams()
        self.ignore_occluded = bool(ignore_occluded)
        self.reseed(seed)
        self._feat_stride = (np.array(geom.image_size_wh, np.float32) - 1.0) / (
            np.array(geom.heatmap_size_wh, np.float32) - 1.0
        )

    def reseed(self, seed: int) -> None:
        """Give this copy of the dataset its own random streams.

        Data-loader workers each get a copy of the dataset; without reseeding
        (see ``seed_worker``) they would all draw the same augmentations.
        """
        self._rng = np.random.default_rng(seed)
        if self._window is not None:
            self._window._rng = np.random.default_rng(self._rng.integers(0, 2**63 - 1))

    def _maybe_flip(
        self, img: np.ndarray, kp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        flipped = False
        if self._fliplr > 0 and self._rng.random() < self._fliplr:
            img, kp = _mirror(img, kp, self._flip_idx, axis=1)
            flipped = True
        if self._flipud > 0 and self._rng.random() < self._flipud:
            img, kp = _mirror(img, kp, self._flip_idx, axis=0)
            flipped = True
        return (np.ascontiguousarray(img) if flipped else img), kp

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        ann, img_meta = self.index[self.ids[i]]
        img = cv2.imread(
            str(image_path(self.dir, img_meta["file_name"])), cv2.IMREAD_COLOR
        )
        kp = np.array(ann["keypoints"], np.float32).reshape(-1, 3)
        # PoseKit images are one-animal canonical crops (single annotation per
        # image; see posekit/gui/main_window.py save_current, which writes one
        # class + keypoint set per frame). The tight per-annotation COCO bbox
        # (compute_bbox_from_kpts, ~3% pad around keypoints) is therefore NOT
        # the crop the model should be centered on at inference time — that's
        # the crop's full extent (infer.py::preprocess_crop uses (0, 0, w, h)).
        # Use the same full-extent box here so box2cs (aspect fix + PADDING_FACTOR)
        # lands on an identical fraction-of-input at train and inference time.
        # ann["bbox"] is left untouched below for PCK normalization (validate.py),
        # which legitimately wants the tight animal extent, not the crop extent.
        kp_warp = kp
        if self._window is not None:
            img, kp_warp = self._window.recrop(img, kp)
        img, kp_warp = self._maybe_flip(img, kp_warp)
        img_h, img_w = img.shape[:2]
        full_box_xywh = np.array([0.0, 0.0, float(img_w), float(img_h)], np.float32)
        center, scale = box2cs(full_box_xywh, geom=self.geom)

        rot = 0.0
        if self.augment:
            if self._legacy_scale:
                scale = scale * float(
                    np.clip(
                        self._rng.normal() * 0.25 + 1.0,
                        1 - _SCALE_JITTER,
                        1 + _SCALE_JITTER,
                    )
                )
            rot = sample_rotation(self.aug, self._rng)

        warped = top_down_affine(img, center, scale, rot, geom=self.geom)
        if self.augment:
            warped = augment_pixels(warped, self.aug, self._rng)
        image = torch.from_numpy(normalize(warped))

        matrix = affine_matrix(center, scale, rot, geom=self.geom)
        joints_in = _warp_joints(kp_warp[:, :2], matrix)  # input-crop space
        joints_hm = joints_in / self._feat_stride  # heatmap space
        vis = kp_warp[:, 2].copy()
        if self.ignore_occluded:
            vis[vis == 1] = 0.0  # COCO v=1: labelled but occluded
        # Zero out visibility (for target/weight generation only) for joints
        # that warp outside the heatmap bounds under augmentation, so
        # JointsMSELoss does not train those channels toward "no keypoint
        # anywhere" on samples where the joint was actually present.
        w_hm, h_hm = self.geom.heatmap_size_wh
        in_bounds = (
            (joints_hm[:, 0] >= 0)
            & (joints_hm[:, 0] <= w_hm - 1)
            & (joints_hm[:, 1] >= 0)
            & (joints_hm[:, 1] <= h_hm - 1)
        )
        vis_eff = vis * in_bounds.astype(vis.dtype)
        target, weight = generate_udp_gaussian(
            joints_hm, vis_eff, self.geom.heatmap_size_wh, self.sigma
        )

        return {
            "image": image,
            "target": torch.from_numpy(target),
            "target_weight": torch.from_numpy(weight),
            "center": torch.from_numpy(center),
            "scale": torch.from_numpy(scale),
            "gt_joints": torch.from_numpy(kp),
            "bbox": torch.tensor(ann["bbox"], dtype=torch.float32),
            "image_id": int(ann["image_id"]),
        }


def flip_index(pairs, num_kpts: int) -> list[int]:
    """Keypoint permutation for a mirror: each left/right pair swaps rows.

    (``RunConfig`` validates the pairs; ``vitpose/`` stays free of
    ``hydra_suite`` imports, so this mirrors ``core.pose_flip.flip_index``.)
    """
    idx = list(range(num_kpts))
    for a, b in pairs or ():
        if 0 <= a < num_kpts and 0 <= b < num_kpts:
            idx[a], idx[b] = idx[b], idx[a]
    return idx


def _mirror(
    img: np.ndarray, kp: np.ndarray, flip_idx: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``img`` along ``axis`` (1 = left-right, 0 = up-down) and the
    keypoints with it; left/right partners swap rows so names stay right."""
    extent = img.shape[1] if axis == 1 else img.shape[0]
    img = np.flip(img, axis=axis)
    kp = kp.copy()
    coord = 0 if axis == 1 else 1
    kp[:, coord] = extent - kp[:, coord]
    return img, kp[flip_idx]


def seed_worker(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=...)``: independent augmentation per worker
    and per epoch (torch hands each worker a fresh base seed every epoch)."""
    import random

    info = torch.utils.data.get_worker_info()
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(seed)
