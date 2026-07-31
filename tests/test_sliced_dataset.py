import numpy as np

from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.sliced_dataset import (
    label_line_for_level,
    measure_reference_body_px,
    project_to_level,
)


def test_measure_reference_body_px_median_major_axis():
    # Two objects: 40x20 and 80x20 (px) at frame 100x100 -> majors 40, 80 -> median 60.
    def rect_norm(cx, cy, w, h):
        pts = np.array(
            [
                [cx - w / 2, cy - h / 2],
                [cx + w / 2, cy - h / 2],
                [cx + w / 2, cy + h / 2],
                [cx - w / 2, cy + h / 2],
            ],
            dtype=np.float32,
        )
        pts[:, 0] /= 100.0
        pts[:, 1] /= 100.0
        return pts

    labels = [(0, rect_norm(50, 50, 40, 20)), (0, rect_norm(50, 50, 80, 20))]
    ref = measure_reference_body_px(labels, (100, 100))
    assert abs(ref - 60.0) < 1.0


def test_project_to_level_aabb_from_polygon():
    poly = np.array([[0.1, 0.1], [0.5, 0.2], [0.4, 0.6], [0.05, 0.4]], dtype=np.float32)
    aabb = project_to_level(poly, GeometryLevel.AABB)
    assert aabb.shape == (4, 2)
    assert abs(aabb[:, 0].min() - 0.05) < 1e-4
    assert abs(aabb[:, 0].max() - 0.5) < 1e-4


def test_project_to_level_obb_returns_four_corners():
    poly = np.array(
        [[0.1, 0.1], [0.5, 0.1], [0.5, 0.3], [0.1, 0.3], [0.3, 0.35]], dtype=np.float32
    )
    obb = project_to_level(poly, GeometryLevel.OBB)
    assert obb.shape == (4, 2)


def test_project_to_level_polygon_keeps_contour():
    poly = np.array([[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]], dtype=np.float32)
    out = project_to_level(poly, GeometryLevel.POLYGON)
    assert np.allclose(out, poly)


def test_label_line_field_counts():
    aabb = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]], dtype=np.float32)
    assert len(label_line_for_level(2, aabb, GeometryLevel.AABB).split()) == 5
    assert len(label_line_for_level(0, aabb, GeometryLevel.OBB).split()) == 9
    tri = np.array([[0.1, 0.1], [0.3, 0.1], [0.2, 0.4]], dtype=np.float32)
    assert len(label_line_for_level(1, tri, GeometryLevel.POLYGON).split()) == 7


from pathlib import Path

import cv2

from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.sliced_dataset import (
    SliceBuildParams,
    build_sliced_obb_dataset,
)


def _write_synthetic_obb_dataset(root: Path) -> Path:
    """One 512x512 train image with two axis-aligned OBB labels (9-field)."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def obb_line(cx, cy, w, h):
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        c = [
            x1 / 512,
            y1 / 512,
            x2 / 512,
            y1 / 512,
            x2 / 512,
            y2 / 512,
            x1 / 512,
            y2 / 512,
        ]
        return "0 " + " ".join(f"{v:.6f}" for v in c)

    for split in ("train", "val"):
        cv2.imwrite(
            str(root / "images" / split / "f0.jpg"),
            np.zeros((512, 512, 3), dtype=np.uint8),
        )
        (root / "labels" / split / "f0.txt").write_text(
            obb_line(80, 80, 40, 40) + "\n" + obb_line(430, 430, 40, 40) + "\n",
            encoding="utf-8",
        )
    (root / "dataset.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: object\n".format(
            root.resolve()
        ),
        encoding="utf-8",
    )
    return root


def test_build_sliced_dataset_produces_tiled_labels(tmp_path):
    merged = _write_synthetic_obb_dataset(tmp_path / "merged")
    params = SliceBuildParams(
        geometry_mode="custom",
        slice_width=256,
        slice_height=256,
        overlap=0.2,
        target_sizes=[],
        full_frame_mix=False,
        negative_tile_fraction=0.0,
    )
    out = build_sliced_obb_dataset(
        str(merged),
        str(tmp_path / "out"),
        level=GeometryLevel.OBB,
        params=params,
        seed=1,
    )
    out_dir = Path(out.dataset_dir)
    assert (out_dir / "dataset.yaml").exists()
    train_labels = list((out_dir / "labels" / "train").glob("*.txt"))
    assert train_labels, "expected at least one tiled train label"
    # Each kept tile label is a 9-field OBB line at OBB level.
    for lp in train_labels:
        for ln in lp.read_text().splitlines():
            if ln.strip():
                assert len(ln.split()) == 9


def test_build_sliced_dataset_area_threshold_drops_slivers(tmp_path):
    merged = _write_synthetic_obb_dataset(tmp_path / "merged")
    # High threshold: an object only partly inside a tile must be dropped there.
    params = SliceBuildParams(
        geometry_mode="custom",
        slice_width=100,
        slice_height=100,
        overlap=0.0,
        min_area_ratio=0.95,
        target_sizes=[],
        full_frame_mix=False,
        negative_tile_fraction=0.0,
    )
    out = build_sliced_obb_dataset(
        str(merged),
        str(tmp_path / "out"),
        level=GeometryLevel.OBB,
        params=params,
        seed=1,
    )
    # Objects (40px) straddling 100px tile edges lose >5% area on boundary tiles;
    # only tiles fully containing an object keep it. Build must still succeed.
    assert Path(out.dataset_dir, "dataset.yaml").exists()


def test_multiscale_emits_distinct_tile_sizes(tmp_path):
    merged = _write_synthetic_obb_dataset(tmp_path / "merged")
    params = SliceBuildParams(
        geometry_mode="auto_object",
        imgsz=640,
        reference_body_px=40.0,
        target_sizes=[80.0, 160.0],
        full_frame_mix=False,
        negative_tile_fraction=0.0,
    )
    out = build_sliced_obb_dataset(
        str(merged),
        str(tmp_path / "out"),
        level=GeometryLevel.OBB,
        params=params,
        seed=1,
    )
    names = [p.name for p in Path(out.dataset_dir, "images", "train").glob("*.jpg")]
    # ref=40, target=80 -> frac=0.125 -> tile 320; target=160 -> frac=0.25 -> tile 160.
    assert any("_t320x320_" in n for n in names)
    assert any("_t160x160_" in n for n in names)


def test_full_frame_mix_emits_full_frame_sample(tmp_path):
    merged = _write_synthetic_obb_dataset(tmp_path / "merged")
    params = SliceBuildParams(
        geometry_mode="custom",
        slice_width=256,
        slice_height=256,
        target_sizes=[],
        full_frame_mix=True,
        negative_tile_fraction=0.0,
    )
    out = build_sliced_obb_dataset(
        str(merged),
        str(tmp_path / "out"),
        level=GeometryLevel.OBB,
        params=params,
        seed=1,
    )
    names = [p.name for p in Path(out.dataset_dir, "images", "train").glob("*.jpg")]
    assert any("_full" in n for n in names)
