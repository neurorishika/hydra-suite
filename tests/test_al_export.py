import json
from collections.abc import Mapping

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


class _SingleReadImages(Mapping):
    """Images mapping that raises if any key is read more than once.

    Proves the whole-export-in-memory fix: `export_al_dataset` must consult
    `images` exactly once per frame (via the authoritative root only) no
    matter how many derived levels are requested, not once per level.
    """

    def __init__(self, frame_ids):
        self._frames = {
            fid: np.zeros((100, 200, 3), dtype=np.uint8) for fid in frame_ids
        }
        self.read_counts: dict[int, int] = {}

    def __getitem__(self, frame_id):
        self.read_counts[frame_id] = self.read_counts.get(frame_id, 0) + 1
        if self.read_counts[frame_id] > 1:
            raise AssertionError(f"frame {frame_id} read more than once")
        return self._frames[frame_id]

    def __iter__(self):
        return iter(self._frames)

    def __len__(self):
        return len(self._frames)


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


def test_images_mapping_is_read_at_most_once_per_frame_across_all_levels(tmp_path):
    """Regression test for the whole-export-in-memory fix.

    `images` must be a lazy Mapping consulted exactly once per frame (by the
    authoritative root only); derived roots must hardlink + read a cached
    shape rather than re-indexing `images`. A mapping that raises on a second
    read of the same key proves no per-level re-read happens even when three
    levels are requested.
    """
    images = _SingleReadImages([0, 1])
    manifest = export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[_frame(0), _frame(1)],
        images=images,
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON, GeometryLevel.OBB, GeometryLevel.AABB],
        class_names=["ant"],
        provenance=_provenance(),
    )
    assert {r["level"] for r in manifest["roots"]} == {"polygon", "obb", "aabb"}
    assert images.read_counts == {0: 1, 1: 1}
    for name in ("polygon", "obb", "aabb"):
        assert (tmp_path / "al_round" / name / "images" / "f000000.jpg").is_file()
        assert (tmp_path / "al_round" / name / "images" / "f000001.jpg").is_file()


# =============================================================================
# FINDING 3: class ids and classes.txt must be reconciled
# =============================================================================


def _multiclass_frame(class_ids, frame_id=0):
    poly = np.array([[10, 20], [30, 20], [30, 40], [10, 45]], dtype=np.float32)
    return ExportedFrame(
        frame_id=frame_id,
        image_name=f"f{frame_id:06d}.jpg",
        records=[
            LabelRecord(
                class_id=cid,
                confidence=0.9,
                points=poly,
                level=GeometryLevel.POLYGON,
            )
            for cid in class_ids
        ],
    )


def test_classes_txt_is_padded_to_cover_every_emitted_class_id(tmp_path):
    """A multi-class checkpoint used to write `3 ...` into a root whose
    classes.txt had a single line, and `_write_root` validated nothing."""
    frames = [_multiclass_frame([0, 3])]
    manifest = export_al_dataset(
        round_dir=tmp_path / "round",
        frames=frames,
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant"],
        provenance=_provenance(),
    )

    classes = (tmp_path / "round" / "polygon" / "classes.txt").read_text().split()
    assert classes == ["ant", "class_1", "class_2", "class_3"]
    assert manifest["class_names"] == classes
    assert manifest["class_names_autofilled"] == ["class_1", "class_2", "class_3"]

    meta = json.loads((tmp_path / "round" / "polygon" / "source.json").read_text())
    assert meta["class_names"] == classes
    assert meta["class_names_autofilled"] == ["class_1", "class_2", "class_3"]

    # Every id in the labels indexes a real line of classes.txt.
    ids = {
        int(line.split()[0])
        for line in (tmp_path / "round" / "polygon" / "labels" / "f000000.txt")
        .read_text()
        .splitlines()
        if line.strip()
    }
    assert ids == {0, 3}
    assert max(ids) < len(classes)


def test_sufficient_class_names_are_left_alone(tmp_path):
    manifest = export_al_dataset(
        round_dir=tmp_path / "round",
        frames=[_multiclass_frame([0, 1])],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant", "larva", "queen"],
        provenance=_provenance(),
    )
    assert manifest["class_names_autofilled"] == []
    assert (tmp_path / "round" / "polygon" / "classes.txt").read_text().split() == [
        "ant",
        "larva",
        "queen",
    ]


# =============================================================================
# FINDING 4: zero-record frames are never written as background samples
# =============================================================================


def _empty_frame(frame_id, is_context=False, drops=None):
    return ExportedFrame(
        frame_id=frame_id,
        image_name=f"f{frame_id:06d}.jpg",
        records=[],
        is_context=is_context,
        drops=drops or {},
    )


def test_zero_record_frames_are_skipped_and_counted(tmp_path):
    manifest = export_al_dataset(
        round_dir=tmp_path / "round",
        frames=[_frame(0), _empty_frame(1, drops={"unmatched": 2}), _frame(2)],
        images=_images([0, 1, 2]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant"],
        provenance=_provenance(),
    )
    labels = tmp_path / "round" / "polygon" / "labels"
    images = tmp_path / "round" / "polygon" / "images"
    assert sorted(p.name for p in labels.iterdir()) == ["f000000.txt", "f000002.txt"]
    assert sorted(p.name for p in images.iterdir()) == ["f000000.jpg", "f000002.jpg"]
    assert manifest["totals"]["frames_exported"] == 2
    assert manifest["totals"]["frames_skipped_no_records"] == 1
    assert manifest["skipped_frame_ids_no_records"] == [1]
    # The skipped frame's drop accounting is not lost.
    assert manifest["totals"]["dropped_unmatched"] == 2
    assert 1 not in manifest["selected_frame_ids"]


def test_a_round_with_no_geometry_at_all_raises(tmp_path):
    with pytest.raises(ValueError, match="no frame in this round"):
        export_al_dataset(
            round_dir=tmp_path / "round",
            frames=[_empty_frame(0), _empty_frame(1)],
            images=_images([0, 1]),
            native_level=GeometryLevel.POLYGON,
            levels=[GeometryLevel.POLYGON],
            class_names=["ant"],
            provenance=_provenance(),
        )
    assert not (tmp_path / "round").exists()
    assert not (tmp_path / "round.partial").exists()


def test_extra_totals_are_merged_into_the_manifest(tmp_path):
    manifest = export_al_dataset(
        round_dir=tmp_path / "round",
        frames=[_frame(0)],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant"],
        provenance=_provenance(),
        extra_totals={"detection_failed": 3},
    )
    assert manifest["totals"]["detection_failed"] == 3


def _degenerate_frame(frame_id, n_points, extra_good=0):
    """A frame whose first record has `n_points` vertices, plus `extra_good` valid ones."""
    records = [
        LabelRecord(
            class_id=0,
            confidence=0.9,
            points=np.array([[10, 20], [30, 25]][:n_points], dtype=np.float32),
            level=GeometryLevel.POLYGON,
        )
    ]
    for _ in range(extra_good):
        records.append(
            LabelRecord(
                class_id=0,
                confidence=0.8,
                points=np.array(
                    [[10, 20], [30, 20], [30, 40], [10, 45]], dtype=np.float32
                ),
                level=GeometryLevel.POLYGON,
            )
        )
    return ExportedFrame(
        frame_id=frame_id,
        image_name=f"f{frame_id:06d}.jpg",
        records=records,
    )


def test_degenerate_contour_drops_its_record_not_the_round(tmp_path):
    """A contour with <3 vertices used to raise from inside write_label_file,
    aborting the whole round and losing every good frame with it. Every other
    per-frame failure here drops-and-counts; this one now matches."""
    frames = [_degenerate_frame(0, n_points=2, extra_good=1), _frame(1)]
    manifest = export_al_dataset(
        round_dir=tmp_path / "round",
        frames=frames,
        images=_images([0, 1]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant"],
        provenance=_provenance(),
    )

    assert manifest["totals"]["dropped_degenerate_geometry"] == 1
    # Both frames survive: frame 0 still had one good record.
    assert manifest["totals"]["frames_exported"] == 2
    assert manifest["totals"]["objects"] == 2
    labels = sorted((tmp_path / "round" / "polygon" / "labels").glob("*.txt"))
    assert len(labels) == 2
    assert len(labels[0].read_text().strip().splitlines()) == 1


def test_frame_emptied_by_degenerate_drops_is_skipped_not_written_empty(tmp_path):
    """Dropping the last record must route into the accounted zero-record skip,
    never write an empty .txt -- YOLO reads that as 'no objects here'."""
    frames = [_degenerate_frame(0, n_points=2), _frame(1)]
    manifest = export_al_dataset(
        round_dir=tmp_path / "round",
        frames=frames,
        images=_images([0, 1]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant"],
        provenance=_provenance(),
    )

    assert manifest["totals"]["dropped_degenerate_geometry"] == 1
    assert manifest["totals"]["frames_skipped_no_records"] == 1
    assert manifest["totals"]["frames_exported"] == 1
    assert manifest["skipped_frame_ids_no_records"] == [0]
    assert manifest["selected_frame_ids"] == [1]
    labels = list((tmp_path / "round" / "polygon" / "labels").glob("*.txt"))
    assert len(labels) == 1
    assert labels[0].read_text().strip() != ""
