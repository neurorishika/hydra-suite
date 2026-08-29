import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.training.contracts import SourceDataset, SplitConfig, TrainingRole
from hydra_suite.training.dataset_builders import (
    _parse_geometry_label_lines,
    blocked_roles_for_level,
    derive_crop_segment_dataset_from_source,
    derive_detect_dataset_from_obb,
    derive_segment_dataset_from_source,
    merge_obb_sources,
    prepare_role_dataset,
    role_min_level,
)
from hydra_suite.training.geometry_levels import GeometryLevel


def test_new_roles_exist():
    assert TrainingRole.DETECT_DIRECT.value == "detect_direct"
    assert TrainingRole.SEGMENT_DIRECT.value == "segment_direct"
    assert TrainingRole.SEQ_CROP_SEGMENT.value == "seq_crop_segment"


def test_parse_polygon_line(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5 0.3 0.7 0.1 0.5\n", encoding="utf-8"
    )  # 5 pts
    parsed = _parse_geometry_label_lines(p)
    assert parsed[0][0] == 0
    assert parsed[0][1].shape == (5, 2)


def test_parse_detect_line_expands_to_quad(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")  # cx cy w h
    cls, pts = _parse_geometry_label_lines(p)[0]
    assert cls == 2 and pts.shape == (4, 2)
    assert np.allclose(pts[0], [0.4, 0.3])  # x1,y1 = cx-w/2, cy-h/2


def _mk_dataset(root: Path, label_line: str):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(root / "images" / split / "a.jpg"), np.zeros((40, 40, 3), np.uint8)
        )
        (root / "labels" / split / "a.txt").write_text(label_line, encoding="utf-8")


def test_detect_from_polygon(tmp_path):
    src = tmp_path / "poly"
    _mk_dataset(src, "0 0.1 0.1 0.9 0.1 0.9 0.9 0.5 0.95 0.1 0.9\n")  # 5-pt contour
    res = derive_detect_dataset_from_obb(src, tmp_path / "out", class_names=["object"])
    line = (
        next((Path(res.dataset_dir) / "labels" / "train").glob("*.txt"))
        .read_text()
        .split()
    )
    assert len(line) == 5  # class + cx cy w h
    assert float(line[3]) == pytest.approx(0.8, abs=1e-3)  # width = 0.9-0.1


def test_segment_passthrough_preserves_points(tmp_path):
    src = tmp_path / "poly"
    line = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.5 0.95 0.1 0.9\n"
    _mk_dataset(src, line)
    res = derive_segment_dataset_from_source(
        src, tmp_path / "out", class_names=["object"]
    )
    out = (
        next((Path(res.dataset_dir) / "labels" / "train").glob("*.txt"))
        .read_text()
        .strip()
    )
    assert len(out.split()) == 11  # class + 5 points preserved


@pytest.mark.parametrize(
    "builder",
    [derive_detect_dataset_from_obb, derive_segment_dataset_from_source],
)
def test_direct_derivations_preserve_slice_geometry(tmp_path, builder):
    src = tmp_path / "sliced"
    _mk_dataset(src, "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
    geometry = {
        "geometry_mode": "auto_object",
        "reference_body_px": 42.0,
        "target_sizes": [200.0, 300.0, 400.0],
    }
    (src / "manifest.json").write_text(
        json.dumps({"type": "sliced_obb", "slice_geometry": geometry}),
        encoding="utf-8",
    )

    result = builder(src, tmp_path / "out", class_names=["object"])
    manifest = json.loads((Path(result.dataset_dir) / "manifest.json").read_text())
    assert manifest["slice_geometry"] == geometry


def test_crop_segment_clips_and_renormalizes(tmp_path):
    src = tmp_path / "poly"
    _mk_dataset(src, "0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6 0.3 0.7\n")
    res = derive_crop_segment_dataset_from_source(
        src,
        tmp_path / "out",
        class_names=["object"],
        enforce_square=False,
        pad_ratio=0.0,
    )
    out = (
        next((Path(res.dataset_dir) / "labels" / "train").glob("*.txt"))
        .read_text()
        .split()
    )
    pts = np.asarray([float(v) for v in out[1:]], dtype=np.float32).reshape(-1, 2)
    assert pts.min() >= 0.0 and pts.max() <= 1.0  # re-normalized into crop space


def test_role_min_levels():
    assert role_min_level(TrainingRole.DETECT_DIRECT) is GeometryLevel.AABB
    assert role_min_level(TrainingRole.OBB_DIRECT) is GeometryLevel.OBB
    assert role_min_level(TrainingRole.SEGMENT_DIRECT) is GeometryLevel.POLYGON
    assert role_min_level(TrainingRole.SEQ_CROP_SEGMENT) is GeometryLevel.POLYGON


def test_blocked_roles_for_aabb_merge():
    roles = [
        TrainingRole.OBB_DIRECT,
        TrainingRole.DETECT_DIRECT,
        TrainingRole.SEGMENT_DIRECT,
    ]
    blocked = blocked_roles_for_level(GeometryLevel.AABB, roles)
    assert TrainingRole.OBB_DIRECT in blocked and TrainingRole.SEGMENT_DIRECT in blocked
    assert TrainingRole.DETECT_DIRECT not in blocked


def test_prepare_segment_direct_refused_above_level(tmp_path):
    with pytest.raises(RuntimeError, match="polygon"):
        prepare_role_dataset(
            TrainingRole.SEGMENT_DIRECT,
            str(tmp_path),
            tmp_path / "out",
            class_names=["object"],
            merged_level=GeometryLevel.OBB,
        )


def test_merged_level_min_and_blocker():
    from hydra_suite.detectkit.gui.dialogs.training_dialog import (
        merged_level_and_blocker,
    )
    from hydra_suite.detectkit.gui.models import OBBSource

    sources = [
        OBBSource(path="/a", name="poly", level="polygon"),
        OBBSource(path="/b", name="boxes", level="obb"),
    ]
    level, blocker = merged_level_and_blocker(sources)
    assert level is GeometryLevel.OBB
    assert blocker is not None and blocker.name == "boxes"


def test_merged_level_empty_is_polygon():
    from hydra_suite.detectkit.gui.dialogs.training_dialog import (
        merged_level_and_blocker,
    )

    level, blocker = merged_level_and_blocker([])
    assert level is GeometryLevel.POLYGON and blocker is None


def test_merge_uses_only_sources_that_can_supply_the_selected_geometry(tmp_path):
    """Mixed project sources are selected per target instead of globally gated."""
    aabb = tmp_path / "aabb"
    polygon = tmp_path / "polygon"
    _mk_dataset(aabb, "0 0.5 0.5 0.4 0.3\n")
    _mk_dataset(
        polygon,
        "0 0.1 0.1 0.8 0.1 0.8 0.7 0.4 0.9 0.1 0.7\n",
    )
    sources = [
        SourceDataset(path=str(aabb), name="boxes", level="aabb"),
        SourceDataset(path=str(polygon), name="masks", level="polygon"),
    ]

    obb = merge_obb_sources(
        sources,
        tmp_path / "out",
        SplitConfig(),
        class_names=["object"],
        dedup=False,
        target_level=GeometryLevel.OBB,
    )
    obb_labels = list((Path(obb.dataset_dir) / "labels").rglob("*.txt"))
    assert obb.stats["source_items"] == {"masks": 2}
    assert all(len(path.read_text().split()) == 9 for path in obb_labels)

    detect = merge_obb_sources(
        sources,
        tmp_path / "out",
        SplitConfig(),
        class_names=["object"],
        dedup=False,
        target_level=GeometryLevel.AABB,
    )
    assert set(detect.stats["source_items"]) == {"boxes", "masks"}
    assert all(
        len(path.read_text().split()) == 5
        for path in (Path(detect.dataset_dir) / "labels").rglob("*.txt")
    )


def test_merge_resolves_the_matching_root_from_a_multilevel_source(tmp_path):
    round_root = tmp_path / "round"
    _mk_dataset(round_root / "obb", "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n")
    _mk_dataset(round_root / "aabb", "0 0.5 0.5 0.7 0.4\n")
    (round_root / "manifest.json").write_text(
        json.dumps(
            {
                "roots": [
                    {"level": "obb", "path": str(round_root / "obb")},
                    {"level": "aabb", "path": str(round_root / "aabb")},
                ]
            }
        ),
        encoding="utf-8",
    )

    merged = merge_obb_sources(
        [SourceDataset(path=str(round_root), name="round", level="obb")],
        tmp_path / "out",
        SplitConfig(),
        class_names=["object"],
        dedup=False,
        target_level=GeometryLevel.AABB,
    )
    labels = list((Path(merged.dataset_dir) / "labels").rglob("*.txt"))
    assert labels
    assert all(len(path.read_text().split()) == 5 for path in labels)


def test_project_roundtrips_detect_segment_model_and_imgsz(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    p = DetectKitProject()
    p.imgsz_detect_direct = 512
    p.imgsz_segment_direct = 768
    p.model_detect_direct = "custom-det.pt"
    p.model_segment_direct = "custom-seg.pt"
    dest = tmp_path / "p.json"
    p.save(dest)
    q = DetectKitProject.load(dest)
    assert q.imgsz_detect_direct == 512 and q.imgsz_segment_direct == 768
    assert (
        q.model_detect_direct == "custom-det.pt"
        and q.model_segment_direct == "custom-seg.pt"
    )
