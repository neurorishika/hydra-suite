from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import HEATMAP_SIZE_WH, IMAGE_SIZE_WH
from ..transforms import affine_matrix, box2cs, normalize, top_down_affine
from .targets import generate_udp_gaussian

FEAT_STRIDE = (np.array(IMAGE_SIZE_WH, np.float32) - 1.0) / (
    np.array(HEATMAP_SIZE_WH, np.float32) - 1.0
)

# augmentation ranges (no flip, per spec)
_SCALE_JITTER = 0.30
_ROT_RANGE = 40.0
_ROT_PROB = 0.6


def load_coco_index(dataset_dir: Path) -> tuple[list[int], dict]:
    coco = json.loads(
        (Path(dataset_dir) / "annotations.json").read_text(encoding="utf-8")
    )
    images = {img["id"]: img for img in coco["images"]}
    anns = [a for a in coco["annotations"] if a.get("num_keypoints", 0) > 0]
    index = {a["id"]: (a, images[a["image_id"]]) for a in anns}
    return list(index.keys()), index


def _warp_joints(joints_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return joints_xy @ matrix[:, :2].T + matrix[:, 2]


class CocoKeypointsDataset(Dataset):
    def __init__(
        self, dataset_dir: Path, ids: list[int], sigma: float, augment: bool
    ) -> None:
        self.dir = Path(dataset_dir)
        self.ids, self.index = load_coco_index(dataset_dir)
        self.ids = [i for i in ids if i in self.index]
        self.sigma = float(sigma)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        ann, img_meta = self.index[self.ids[i]]
        img = cv2.imread(
            str(self.dir / "images" / img_meta["file_name"]), cv2.IMREAD_COLOR
        )
        kp = np.array(ann["keypoints"], np.float32).reshape(-1, 3)
        center, scale = box2cs(np.array(ann["bbox"], np.float32))

        rot = 0.0
        if self.augment:
            scale = scale * float(
                np.clip(
                    np.random.randn() * 0.25 + 1.0, 1 - _SCALE_JITTER, 1 + _SCALE_JITTER
                )
            )
            if np.random.rand() < _ROT_PROB:
                rot = float(
                    np.clip(
                        np.random.randn() * (_ROT_RANGE / 2), -_ROT_RANGE, _ROT_RANGE
                    )
                )

        warped = top_down_affine(img, center, scale, rot)
        if self.augment:
            warped = _photometric(warped)
        image = torch.from_numpy(normalize(warped))

        matrix = affine_matrix(center, scale, rot)
        joints_in = _warp_joints(kp[:, :2], matrix)  # input-crop space
        joints_hm = joints_in / FEAT_STRIDE  # heatmap space
        vis = kp[:, 2]
        target, weight = generate_udp_gaussian(
            joints_hm, vis, HEATMAP_SIZE_WH, self.sigma
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


def _photometric(img_bgr: np.ndarray) -> np.ndarray:
    out = img_bgr.astype(np.float32)
    out *= np.random.uniform(0.7, 1.3)  # brightness
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * np.random.uniform(0.7, 1.3) + mean  # contrast
    return np.clip(out, 0, 255).astype(np.uint8)
