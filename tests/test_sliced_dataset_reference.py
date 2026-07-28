from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.sliced_dataset import (
    SliceBuildParams,
    build_sliced_obb_dataset,
    object_major_axes_px,
)


def _rect_norm(cx, cy, w, h, W=512, H=512):
    pts = np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ],
        dtype=np.float32,
    )
    pts[:, 0] /= W
    pts[:, 1] /= H
    return pts


def test_object_major_axes_px_returns_all_majors():
    labels = [(0, _rect_norm(100, 100, 40, 20)), (0, _rect_norm(200, 200, 80, 20))]
    majors = object_major_axes_px(labels, (512, 512))
    assert sorted(round(m) for m in majors) == [40, 80]


def _write_dataset(root: Path, majors_px):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def obb_line(cx, cy, w, h):
        p = _rect_norm(cx, cy, w, h)
        return "0 " + " ".join(f"{v:.6f}" for v in p.reshape(-1))

    for split in ("train", "val"):
        cv2.imwrite(
            str(root / "images" / split / "f0.jpg"), np.zeros((512, 512, 3), np.uint8)
        )
        # two objects of the given major sizes (square, so major == side)
        lines = [
            obb_line(120, 120, majors_px[0], majors_px[0]),
            obb_line(360, 360, majors_px[1], majors_px[1]),
        ]
        (root / "labels" / split / "f0.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    (root / "dataset.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: object\n",
        encoding="utf-8",
    )
    return root


def test_manifest_records_measured_reference_when_params_zero(tmp_path):
    merged = _write_dataset(tmp_path / "merged", majors_px=(40, 80))
    params = SliceBuildParams(
        geometry_mode="custom",
        slice_width=256,
        slice_height=256,
        target_sizes=[],
        full_frame_mix=False,
        negative_tile_fraction=0.0,
        reference_body_px=0.0,
    )
    out = build_sliced_obb_dataset(
        str(merged),
        str(tmp_path / "out"),
        level=GeometryLevel.OBB,
        params=params,
        seed=1,
    )
    # 4 objects across 2 frames: majors 40,80,40,80 -> median 60.
    assert abs(out.stats["measured_reference_body_px"] - 60.0) < 1.0
    assert abs(out.stats["slice_geometry"]["reference_body_px"] - 60.0) < 1.0


def test_manifest_records_explicit_reference_when_params_set(tmp_path):
    merged = _write_dataset(tmp_path / "merged", majors_px=(40, 80))
    params = SliceBuildParams(
        geometry_mode="custom",
        slice_width=256,
        slice_height=256,
        target_sizes=[],
        full_frame_mix=False,
        negative_tile_fraction=0.0,
        reference_body_px=123.0,
    )
    out = build_sliced_obb_dataset(
        str(merged),
        str(tmp_path / "out"),
        level=GeometryLevel.OBB,
        params=params,
        seed=1,
    )
    assert out.stats["slice_geometry"]["reference_body_px"] == 123.0
    # measured is still reported (independent of the explicit override).
    assert out.stats["measured_reference_body_px"] > 0.0
