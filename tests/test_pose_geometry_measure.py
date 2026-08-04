"""Measuring a PoseKit label set to suggest a ViTPose input geometry."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from hydra_suite.training.pose_geometry_measure import (
    MAX_SUGGESTED_SIZE,
    PoseSizeStats,
    measure_pose_geometry,
)

K = 4  # keypoints per instance in these fixtures


def _write_frame(tmp_path, name, img_wh, instances, k=K):
    """One image + one YOLO-pose label file.

    instances: list of (x0, y0, x1, y1, visible) in PIXELS. Each becomes one
    label line whose k keypoints are placed at the box corners, so the visible
    extent is exactly the box.
    """
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    w_px, h_px = img_wh
    img_path = images / f"{name}.png"
    Image.new("RGB", (w_px, h_px), (0, 0, 0)).save(img_path)

    lines = []
    for x0, y0, x1, y1, visible in instances:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)][:k]
        parts = ["0", "0.5", "0.5", "0.5", "0.5"]
        for cx, cy in corners:
            parts += [f"{cx / w_px:.6f}", f"{cy / h_px:.6f}", "2" if visible else "0"]
        lines.append(" ".join(parts))
    (labels / f"{name}.txt").write_text("\n".join(lines) + "\n")
    return img_path, labels


def test_square_animals_give_a_square_suggestion(tmp_path):
    paths = []
    for i in range(5):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(100, 100, 228, 228, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    assert isinstance(stats, PoseSizeStats)
    assert stats.sample_count == 5
    assert stats.median_aspect == pytest.approx(1.0)
    assert stats.median_long_px == pytest.approx(128.0)
    assert stats.suggested_hw == [128, 128]
    assert stats.clamped is False


def test_wide_animals_give_width_greater_than_height(tmp_path):
    paths = []
    for i in range(5):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(50, 100, 178, 164, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    # box is 128 wide, 64 tall -> aspect 2.0
    assert stats.median_aspect == pytest.approx(2.0)
    h, w = stats.suggested_hw
    assert w > h
    assert stats.suggested_hw == [64, 128]


def test_tall_animals_give_height_greater_than_width(tmp_path):
    paths = []
    for i in range(5):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(100, 50, 164, 178, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    # box is 64 wide, 128 tall -> aspect 0.5
    assert stats.median_aspect == pytest.approx(0.5)
    h, w = stats.suggested_hw
    assert h > w
    assert stats.suggested_hw == [128, 64]


def test_every_instance_on_a_multi_animal_frame_is_counted(tmp_path):
    # PoseKit's own reader parses only the first line; this module must not.
    p, labels = _write_frame(
        tmp_path,
        "multi",
        (400, 400),
        [
            (0, 0, 128, 128, True),
            (200, 0, 328, 128, True),
            (0, 200, 128, 328, True),
        ],
    )
    stats = measure_pose_geometry([p], labels, K)
    assert stats.sample_count == 3
    assert stats.frames_scanned == 1


def test_invisible_keypoints_are_excluded_from_the_extent(tmp_path):
    # Two visible corners 128 apart, two invisible ones far away. If the
    # invisible pair were counted the extent would be much larger.
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (400, 400), (0, 0, 0)).save(images / "a.png")
    parts = ["0", "0.5", "0.5", "0.5", "0.5"]
    for cx, cy, v in [(100, 100, 2), (228, 228, 2), (0, 0, 0), (399, 399, 0)]:
        parts += [f"{cx / 400:.6f}", f"{cy / 400:.6f}", str(v)]
    (labels / "a.txt").write_text(" ".join(parts) + "\n")
    stats = measure_pose_geometry([images / "a.png"], labels, K)
    assert stats.median_long_px == pytest.approx(128.0)


def test_detail_multiplier_scales_the_suggestion(tmp_path):
    paths = []
    for i in range(3):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(100, 100, 228, 228, True)]
        )
        paths.append(p)
    assert measure_pose_geometry(paths, labels, K, detail=2.0).suggested_hw == [
        256,
        256,
    ]
    assert measure_pose_geometry(paths, labels, K, detail=0.5).suggested_hw == [64, 64]


def test_suggestions_are_always_multiples_of_thirty_two(tmp_path):
    paths = []
    for i in range(3):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (500, 500), [(10, 10, 157, 123, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    assert all(v % 32 == 0 for v in stats.suggested_hw)


def test_very_large_animals_are_clamped_and_flagged(tmp_path):
    paths = []
    for i in range(3):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (2000, 2000), [(100, 100, 1700, 1700, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    assert stats.clamped is True
    assert max(stats.suggested_hw) == MAX_SUGGESTED_SIZE


def test_p90_is_reported_and_at_least_the_median(tmp_path):
    paths = []
    for i in range(9):
        extent = 64 if i < 8 else 320  # one large outlier
        p, labels = _write_frame(
            tmp_path, f"f{i}", (600, 600), [(10, 10, 10 + extent, 10 + extent, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    # abs tolerance (not the default 1e-6 rel) to absorb the fixture's own
    # float round-trip: 10/600 and 74/600 don't terminate in 6 decimals, so
    # `_write_frame`'s "{:.6f}" quantizes them and the module's re-multiply
    # by w_px reproduces 63.9996..., not exactly 64.0. Verified by hand:
    # fx0*w=10.0002, fx1*w=73.9998 -> bw=63.9996. Not a `_snap`/aspect/pixel
    # normalization bug -- the module correctly reads what the fixture wrote.
    assert stats.median_long_px == pytest.approx(64.0, abs=1e-3)
    assert stats.p90_long_px > stats.median_long_px


def test_measurement_is_deterministic(tmp_path):
    paths = []
    for i in range(30):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(i, i, i + 100, i + 120, True)]
        )
        paths.append(p)
    a = measure_pose_geometry(paths, labels, K, max_images=10)
    b = measure_pose_geometry(paths, labels, K, max_images=10)
    assert a == b


def test_subsampling_caps_the_frames_scanned(tmp_path):
    paths = []
    for i in range(25):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(10, 10, 138, 138, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K, max_images=7)
    assert stats.frames_scanned == 7


def test_unreadable_image_is_skipped_not_fatal(tmp_path):
    good, labels = _write_frame(
        tmp_path, "good", (400, 400), [(100, 100, 228, 228, True)]
    )
    bad = tmp_path / "images" / "bad.png"
    bad.write_bytes(b"not an image")
    (labels / "bad.txt").write_text((labels / "good.txt").read_text())
    stats = measure_pose_geometry([good, bad], labels, K)
    assert stats.frames_scanned == 2
    assert stats.frames_skipped == 1
    assert stats.sample_count == 1


def test_short_label_line_is_skipped(tmp_path):
    good, labels = _write_frame(
        tmp_path, "good", (400, 400), [(100, 100, 228, 228, True)]
    )
    with (labels / "good.txt").open("a", encoding="utf-8") as fh:
        fh.write("0 0.5 0.5 0.5 0.5 0.1 0.1 2\n")  # only 1 keypoint, needs 4
    stats = measure_pose_geometry([good], labels, K)
    assert stats.sample_count == 1


def test_no_labelled_frames_raises(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(images / "a.png")
    with pytest.raises(ValueError, match="no labelled frames"):
        measure_pose_geometry([images / "a.png"], labels, K)


def test_labels_present_but_none_usable_raises(tmp_path):
    # every keypoint invisible -> no usable instance
    p, labels = _write_frame(tmp_path, "a", (400, 400), [(100, 100, 228, 228, False)])
    with pytest.raises(ValueError, match="usable"):
        measure_pose_geometry([p], labels, K)


def test_non_positive_detail_raises(tmp_path):
    p, labels = _write_frame(tmp_path, "a", (400, 400), [(100, 100, 228, 228, True)])
    with pytest.raises(ValueError, match="detail"):
        measure_pose_geometry([p], labels, K, detail=0.0)


def test_module_imports_no_app_layer():
    """Training must not depend on any app layer -- the reason this module
    re-implements label parsing instead of reusing PoseKit's reader."""
    import ast

    import hydra_suite.training.pose_geometry_measure as mod

    app_packages = {
        "posekit",
        "classkit",
        "detectkit",
        "trackerkit",
        "refinekit",
        "filterkit",
    }
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    offenders = [
        name for name in imported if any(pkg in name.split(".") for pkg in app_packages)
    ]
    assert not offenders, f"training module imports app layers: {offenders}"
