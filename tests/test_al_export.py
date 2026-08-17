import json

import numpy as np
import pytest

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.export import ExportedFrame, export_al_dataset
from hydra_suite.utils.geometry_levels import GeometryLevel


def _frame(frame_id=0, is_context=False, drops=None):
    poly = np.array([[10, 20], [30, 20], [30, 40], [10, 45]], dtype=np.float32)
    return ExportedFrame(
        frame_id=frame_id,
        image_name=f"f{frame_id:06d}.jpg",
        records=[
            LabelRecord(
                class_id=0,
                confidence=0.9,
                points=poly,
                level=GeometryLevel.POLYGON,
            )
        ],
        is_context=is_context,
        drops=drops or {},
    )


def _images(frame_ids):
    return {fid: np.zeros((100, 200, 3), dtype=np.uint8) for fid in frame_ids}


def _provenance():
    return {
        "model_path": "seg.pt",
        "model_task": "segment",
        "preset": "tracker_default",
    }


def test_writes_one_root_per_requested_level(tmp_path):
    manifest = export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[_frame(0)],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON, GeometryLevel.OBB, GeometryLevel.AABB],
        class_names=["ant"],
        provenance=_provenance(),
    )
    root = tmp_path / "al_round"
    for name in ("polygon", "obb", "aabb"):
        assert (root / name / "images" / "f000000.jpg").is_file()
        assert (root / name / "labels" / "f000000.txt").is_file()
        assert (root / name / "classes.txt").read_text() == "ant\n"
        assert (root / name / "source.json").is_file()
    assert {r["level"] for r in manifest["roots"]} == {"polygon", "obb", "aabb"}


def test_authoritative_root_is_the_native_level(tmp_path):
    export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[_frame(0)],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON, GeometryLevel.OBB],
        class_names=["ant"],
        provenance=_provenance(),
    )
    root = tmp_path / "al_round"
    poly_meta = json.loads((root / "polygon" / "source.json").read_text())
    obb_meta = json.loads((root / "obb" / "source.json").read_text())
    assert poly_meta["authoritative"] is True
    assert poly_meta["derived_from"] is None
    assert poly_meta["reviewed"] is True
    assert obb_meta["authoritative"] is False
    assert obb_meta["derived_from"] == "polygon"
    assert obb_meta["reviewed"] is False


def test_derived_root_images_are_hardlinks(tmp_path):
    export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[_frame(0)],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON, GeometryLevel.OBB],
        class_names=["ant"],
        provenance=_provenance(),
    )
    root = tmp_path / "al_round"
    a = (root / "polygon" / "images" / "f000000.jpg").stat()
    b = (root / "obb" / "images" / "f000000.jpg").stat()
    assert a.st_ino == b.st_ino


def test_refuses_level_above_native(tmp_path):
    with pytest.raises(ValueError, match="upward|not achievable"):
        export_al_dataset(
            round_dir=tmp_path / "al_round",
            frames=[_frame(0)],
            images=_images([0]),
            native_level=GeometryLevel.OBB,
            levels=[GeometryLevel.POLYGON],
            class_names=["ant"],
            provenance=_provenance(),
        )


def test_manifest_records_drop_accounting_and_context(tmp_path):
    manifest = export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[
            _frame(0, drops={"lost": 2, "unmatched": 1}),
            _frame(5, is_context=True, drops={"lost": 0, "unmatched": 3}),
        ],
        images=_images([0, 5]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant"],
        provenance=_provenance(),
    )
    assert manifest["totals"]["dropped_lost"] == 2
    assert manifest["totals"]["dropped_unmatched"] == 4
    assert manifest["totals"]["frames_exported"] == 2
    assert manifest["selected_frame_ids"] == [0]
    assert manifest["context_frame_ids"] == [5]


def test_provenance_is_stamped_into_source_json(tmp_path):
    export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[_frame(0)],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant"],
        provenance=_provenance(),
    )
    meta = json.loads((tmp_path / "al_round" / "polygon" / "source.json").read_text())
    assert meta["provenance"]["model_task"] == "segment"
    assert meta["source_kind"] == "trackerkit_al"


def test_partial_write_is_not_left_behind_on_failure(tmp_path, monkeypatch):
    import hydra_suite.data.al.export as export_mod

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(export_mod, "write_label_file", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        export_al_dataset(
            round_dir=tmp_path / "al_round",
            frames=[_frame(0)],
            images=_images([0]),
            native_level=GeometryLevel.POLYGON,
            levels=[GeometryLevel.POLYGON],
            class_names=["ant"],
            provenance=_provenance(),
        )
    assert not (tmp_path / "al_round").exists()
