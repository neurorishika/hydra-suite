"""Tests for DetectKit external-source standardization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hydra_suite.detectkit.gui.source_import import (
    IMPORT_MODE_LINKED,
    IMPORT_MODE_PORTABLE,
    inspect_detectkit_source,
    materialize_detectkit_source,
    resolve_al_round_authoritative_level,
)


def _write_fake_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")


def test_inspect_detectkit_source_accepts_existing_canonical_root(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("ant\n", encoding="utf-8")

    inspection = inspect_detectkit_source(tmp_path)

    assert inspection.source_kind == "detectkit"
    assert inspection.requires_import is False
    assert inspection.discovered_labels == ["ant"]


def test_portable_mode_copies_an_existing_canonical_source(tmp_path: Path):
    source_root = tmp_path / "canonical"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    _write_fake_image(source_root / "images" / "frame001.jpg")
    (source_root / "labels" / "frame001.txt").write_text(
        "0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n", encoding="utf-8"
    )
    (source_root / "classes.txt").write_text("ant\n", encoding="utf-8")

    materialized = materialize_detectkit_source(
        source_root,
        tmp_path / "project",
        import_mode=IMPORT_MODE_PORTABLE,
    )

    assert materialized.imported is True
    assert materialized.canonical_path != source_root.resolve()
    assert materialized.canonical_path.is_relative_to((tmp_path / "project").resolve())
    assert (materialized.canonical_path / "images" / "frame001.jpg").exists()


def test_linked_mode_remains_linked_when_remapping_forced(tmp_path: Path):
    source_root = tmp_path / "canonical"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    _write_fake_image(source_root / "images" / "frame001.jpg")
    (source_root / "labels" / "frame001.txt").write_text(
        "0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n", encoding="utf-8"
    )
    (source_root / "classes.txt").write_text("ant\n", encoding="utf-8")

    materialized = materialize_detectkit_source(
        source_root,
        tmp_path / "project",
        import_mode=IMPORT_MODE_LINKED,
        force_import=True,
    )

    assert materialized.imported is False
    assert materialized.canonical_path == source_root.resolve()


def test_materialize_detectkit_source_converts_yolo_detect_boxes(tmp_path: Path):
    source_root = tmp_path / "yolo_detect"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    _write_fake_image(source_root / "images" / "frame001.jpg")
    (source_root / "labels" / "frame001.txt").write_text(
        "0 0.5 0.5 0.4 0.2\n",
        encoding="utf-8",
    )
    (source_root / "dataset.yaml").write_text(
        "train: images\nnames:\n  0: ant\n",
        encoding="utf-8",
    )

    project_dir = tmp_path / "project"
    materialized = materialize_detectkit_source(source_root, project_dir)

    assert materialized.imported is True
    assert materialized.source_kind == "yolo_detect"
    assert materialized.source_root == source_root.resolve()

    classes_txt = (materialized.canonical_path / "classes.txt").read_text(
        encoding="utf-8"
    )
    label_text = (materialized.canonical_path / "labels" / "frame001.txt").read_text(
        encoding="utf-8"
    )
    fields = label_text.strip().split()

    assert classes_txt == "ant\n"
    assert len(fields) == 9
    assert fields[0] == "0"


def test_materialize_detectkit_source_converts_coco_bbox_annotations(tmp_path: Path):
    source_root = tmp_path / "coco"
    _write_fake_image(source_root / "images" / "sample.jpg")
    (source_root / "annotations").mkdir(parents=True, exist_ok=True)
    (source_root / "annotations" / "instances.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 7,
                        "file_name": "sample.jpg",
                        "width": 100,
                        "height": 50,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 7,
                        "category_id": 5,
                        "bbox": [10, 5, 40, 20],
                    }
                ],
                "categories": [{"id": 5, "name": "ant"}],
            }
        ),
        encoding="utf-8",
    )

    materialized = materialize_detectkit_source(source_root, tmp_path / "project")

    assert materialized.imported is True
    assert materialized.source_kind == "coco"
    assert (materialized.canonical_path / "images" / "sample.jpg").exists()
    assert (materialized.canonical_path / "classes.txt").read_text(
        encoding="utf-8"
    ) == "ant\n"

    fields = (
        (materialized.canonical_path / "labels" / "sample.txt")
        .read_text(encoding="utf-8")
        .strip()
        .split()
    )
    assert len(fields) == 9
    assert fields[0] == "0"


def test_coco_multipolygon_uses_one_valid_component_instead_of_bridging(
    tmp_path: Path,
):
    source_root = tmp_path / "coco"
    _write_fake_image(source_root / "images" / "sample.jpg")
    (source_root / "annotations.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [
                            [1, 1, 4, 1, 4, 4, 1, 4],
                            [50, 50, 90, 50, 90, 90, 50, 90],
                        ],
                    }
                ],
                "categories": [{"id": 1, "name": "ant"}],
            }
        ),
        encoding="utf-8",
    )

    materialized = materialize_detectkit_source(source_root, tmp_path / "project")
    fields = (
        (materialized.canonical_path / "labels" / "sample.txt")
        .read_text(encoding="utf-8")
        .strip()
        .split()
    )

    assert len(fields) == 9
    coords = [float(value) for value in fields[1:]]
    assert min(coords) == pytest.approx(0.5)
    assert max(coords) == pytest.approx(0.9)


def _rle_counts(mask) -> list[int]:
    flat = mask.flatten(order="F")
    counts: list[int] = []
    value = 0
    run = 0
    for pixel in flat:
        pixel_value = int(pixel)
        if pixel_value == value:
            run += 1
        else:
            counts.append(run)
            run = 1
            value = pixel_value
    counts.append(run)
    return counts


def _compress_rle_counts(counts: list[int]) -> str:
    encoded: list[str] = []
    for index, original in enumerate(counts):
        value = int(original)
        if index > 2:
            value -= counts[index - 2]
        while True:
            char = value & 0x1F
            value >>= 5
            more = value != (-1 if char & 0x10 else 0)
            if more:
                char |= 0x20
            encoded.append(chr(char + 48))
            if not more:
                break
    return "".join(encoded)


@pytest.mark.parametrize("compressed", [False, True])
def test_coco_rle_segmentation_is_converted_to_polygon(
    tmp_path: Path, compressed: bool
):
    import numpy as np

    source_root = tmp_path / "coco"
    _write_fake_image(source_root / "images" / "sample.jpg")
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:8, 3:10] = 1
    counts = _rle_counts(mask)
    segmentation = {
        "size": [10, 12],
        "counts": _compress_rle_counts(counts) if compressed else counts,
    }
    (source_root / "annotations.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "sample.jpg", "width": 12, "height": 10}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": segmentation,
                        "bbox": [3, 2, 7, 6],
                    }
                ],
                "categories": [{"id": 1, "name": "ant"}],
            }
        ),
        encoding="utf-8",
    )

    materialized = materialize_detectkit_source(source_root, tmp_path / "project")
    fields = (
        (materialized.canonical_path / "labels" / "sample.txt")
        .read_text(encoding="utf-8")
        .strip()
        .split()
    )

    assert materialized.level == "polygon"
    assert len(fields) >= 7
    assert len(fields) % 2 == 1


def _write_al_round(
    round_dir: Path,
    *,
    stale_paths_root: Path | None = None,
    levels: tuple[tuple[str, bool], ...] = (("obb", True), ("aabb", False)),
) -> None:
    """Write an AL round container: manifest.json + obb/ (authoritative) + aabb/
    (derived) sibling roots, mirroring hydra_suite.data.al.export.export_al_dataset.

    *stale_paths_root*, if given, makes the manifest record each root's path
    under that (nonexistent) location instead of under *round_dir* -- as if
    the round had been copied/renamed after export -- so callers must fall
    back to ``round_dir / level`` to find the real data.
    """
    for level, _authoritative in levels:
        level_dir = round_dir / level
        _write_fake_image(level_dir / "images" / "f001.jpg")
        (level_dir / "labels").mkdir(parents=True, exist_ok=True)
        (level_dir / "labels" / "f001.txt").write_text(
            "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
        )
        (level_dir / "classes.txt").write_text("ant\n", encoding="utf-8")

    manifest_root = stale_paths_root if stale_paths_root is not None else round_dir
    manifest = {
        "schema_version": 2,
        "round_dir": str(round_dir),
        "native_level": "obb",
        "roots": [
            {
                "level": level,
                "authoritative": authoritative,
                "derived_from": None if authoritative else "obb",
                "reviewed": authoritative,
                "path": str(manifest_root / level),
            }
            for level, authoritative in levels
        ],
        "class_names": ["ant"],
    }
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_resolve_al_round_authoritative_level_reads_manifest(tmp_path: Path):
    round_dir = tmp_path / "active_learning" / "20260827_172624"
    _write_al_round(
        round_dir,
        levels=(("aabb", True), ("obb", False)),
    )

    assert resolve_al_round_authoritative_level(round_dir) == "aabb"


def test_resolve_al_round_authoritative_level_falls_back_to_first_root(
    tmp_path: Path,
):
    """No entry flagged authoritative -> use the FIRST entry's declared level.

    `_select_al_round_authoritative_root` already falls back to ``roots[0]``
    for the path in exactly this case. Returning None here instead was
    indistinguishable to `_add_source` from "not an AL round at all", so it
    fell back to the unreliable re-scanned level guess -- registering a
    source whose path came from roots[0] but whose level came from a re-scan
    that cannot tell an axis-aligned-quad AABB from a genuine OBB.
    """
    round_dir = tmp_path / "active_learning" / "20260827_172624"
    _write_al_round(round_dir, levels=(("aabb", False), ("obb", False)))

    assert resolve_al_round_authoritative_level(round_dir) == "aabb"


def test_resolve_al_round_authoritative_level_none_for_non_al_round(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("ant\n", encoding="utf-8")

    assert resolve_al_round_authoritative_level(tmp_path) is None


def test_inspect_detectkit_source_resolves_al_round_to_authoritative_root(
    tmp_path: Path,
):
    round_dir = tmp_path / "active_learning" / "20260827_172624"
    _write_al_round(round_dir)

    inspection = inspect_detectkit_source(round_dir)

    assert inspection.dataset_root == (round_dir / "obb").resolve()
    assert inspection.source_kind == "detectkit_al"
    assert inspection.discovered_labels == ["ant"]


def test_materialize_detectkit_source_imports_al_round_authoritative_root(
    tmp_path: Path,
):
    round_dir = tmp_path / "active_learning" / "20260827_172624"
    _write_al_round(round_dir)

    materialized = materialize_detectkit_source(round_dir, tmp_path / "project")

    assert materialized.source_root == (round_dir / "obb").resolve()
    assert (materialized.canonical_path / "images" / "f001.jpg").exists()
    assert (materialized.canonical_path / "classes.txt").read_text(
        encoding="utf-8"
    ) == "ant\n"


def test_inspect_detectkit_source_falls_back_when_manifest_paths_are_stale(
    tmp_path: Path,
):
    round_dir = tmp_path / "moved" / "20260827_172624"
    stale_root = tmp_path / "original_location_no_longer_exists"
    _write_al_round(round_dir, stale_paths_root=stale_root)

    inspection = inspect_detectkit_source(round_dir)

    assert inspection.dataset_root == (round_dir / "obb").resolve()
    assert inspection.source_kind == "detectkit_al"


def test_inspect_detectkit_source_raises_when_authoritative_root_missing(
    tmp_path: Path,
):
    """If the authoritative root is gone (deleted, and its manifest path is
    also stale/unresolvable) but a derived sibling survives, the single-root
    redirect must refuse rather than silently presenting the unreviewed
    derived sibling as if it were the whole round."""
    round_dir = tmp_path / "active_learning" / "20260827_172624"
    _write_al_round(round_dir)
    shutil.rmtree(round_dir / "obb")

    with pytest.raises(ValueError):
        inspect_detectkit_source(round_dir)


def test_materialize_detectkit_source_can_link_and_normalize_in_place(tmp_path: Path):
    source_root = tmp_path / "linked_yolo_detect"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    _write_fake_image(source_root / "images" / "frame001.jpg")
    (source_root / "labels" / "frame001.txt").write_text(
        "0 0.5 0.5 0.4 0.2\n",
        encoding="utf-8",
    )
    (source_root / "dataset.yaml").write_text(
        "train: images\nnames:\n  0: ant\n",
        encoding="utf-8",
    )

    materialized = materialize_detectkit_source(
        source_root,
        tmp_path / "project",
        import_mode=IMPORT_MODE_LINKED,
    )

    assert materialized.imported is False
    assert materialized.canonical_path == source_root.resolve()
    assert (source_root / "classes.txt").read_text(encoding="utf-8") == "ant\n"
    fields = (
        (source_root / "labels" / "frame001.txt")
        .read_text(encoding="utf-8")
        .strip()
        .split()
    )
    assert len(fields) == 9
    assert fields[0] == "0"


def test_materialize_detectkit_source_preserves_polygon_labels(tmp_path: Path):
    """A polygon-level source must materialize as polygons, not collapse to quads.

    Regression: `_convert_yolo_label_text` accepted only 5- or 9-field lines,
    so making a project portable from a polygon-level source (e.g. an AL round
    whose labels were escalated to masks) raised
    "Unsupported YOLO annotation format ... expected 5 or 9 fields".
    """
    source_root = tmp_path / "polygon_source"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    _write_fake_image(source_root / "images" / "frame001.jpg")
    polygon_line = "0 0.1 0.1 0.5 0.05 0.9 0.1 0.9 0.9 0.1 0.9"
    (source_root / "labels" / "frame001.txt").write_text(
        polygon_line + "\n", encoding="utf-8"
    )
    (source_root / "classes.txt").write_text("ant\n", encoding="utf-8")

    materialized = materialize_detectkit_source(
        source_root,
        tmp_path / "project",
        import_mode=IMPORT_MODE_PORTABLE,
    )

    assert materialized.level == "polygon"
    label_text = (materialized.canonical_path / "labels" / "frame001.txt").read_text(
        encoding="utf-8"
    )
    fields = label_text.strip().split()
    assert len(fields) == 11
    assert fields[0] == "0"
    assert [float(value) for value in fields[1:]] == [
        float(value) for value in polygon_line.split()[1:]
    ]


def test_materialize_detectkit_source_rejects_malformed_label_lines(tmp_path: Path):
    source_root = tmp_path / "bad_source"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    _write_fake_image(source_root / "images" / "frame001.jpg")
    (source_root / "labels" / "frame001.txt").write_text(
        "0 0.1 0.2 0.3\n", encoding="utf-8"
    )
    (source_root / "classes.txt").write_text("ant\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsupported YOLO annotation format"):
        materialize_detectkit_source(
            source_root,
            tmp_path / "project",
            import_mode=IMPORT_MODE_PORTABLE,
        )
