"""Tests for dataset panel utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from hydra_suite.detectkit.gui.utils import (
    ensure_detectkit_source_structure,
    find_label_for_image,
    list_images_in_source,
    parse_obb_label,
    source_class_id_map,
    source_has_images,
)


def test_list_images_in_source_with_images_dir(tmp_path: Path):
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    (img_dir / "a.jpg").write_text("fake")
    (img_dir / "b.png").write_text("fake")
    (img_dir / "c.txt").write_text("not an image")
    images = list_images_in_source(str(tmp_path))
    assert len(images) == 2
    assert source_has_images(str(tmp_path)) is True


def test_source_has_images_does_not_require_a_materialized_listing(tmp_path: Path):
    (tmp_path / "images" / "nested").mkdir(parents=True)
    (tmp_path / "images" / "nested" / "notes.txt").write_text("not an image")
    assert source_has_images(str(tmp_path)) is False


def test_find_label_for_image(tmp_path: Path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    img = tmp_path / "images" / "train" / "frame001.jpg"
    img.write_text("fake")
    lbl = tmp_path / "labels" / "train" / "frame001.txt"
    lbl.write_text("0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8")
    result = find_label_for_image(img, str(tmp_path))
    assert result is not None
    assert result.name == "frame001.txt"


def test_find_label_for_image_does_not_cross_split_stem_collision(tmp_path: Path):
    """An unlabeled train image must not inherit a same-stem val label."""
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    train_image = tmp_path / "images" / "train" / "frame001.jpg"
    train_image.write_bytes(b"fake")
    (tmp_path / "images" / "val" / "frame001.jpg").write_bytes(b"fake")
    (tmp_path / "labels" / "val" / "frame001.txt").write_text(
        "0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n"
    )

    assert find_label_for_image(train_image, str(tmp_path)) is None


def test_detectkit_source_structure_requires_images_labels_and_classes(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()

    with pytest.raises(RuntimeError, match="classes.txt"):
        ensure_detectkit_source_structure(tmp_path)


def test_source_class_id_map_accepts_source_superset(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("bee\nant\nwasp\n", encoding="utf-8")

    class_id_map = source_class_id_map(tmp_path, ["ant", "bee"])

    assert class_id_map == {1: 0, 0: 1}


def test_parse_obb_label_filters_and_remaps_by_project_classes(tmp_path: Path):
    lbl = tmp_path / "filtered.txt"
    lbl.write_text(
        "1 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n" "2 0.2 0.2 0.8 0.2 0.8 0.7 0.2 0.7\n",
        encoding="utf-8",
    )

    dets = parse_obb_label(lbl, img_w=100, img_h=100, class_id_map={1: 0})

    assert [det["class_id"] for det in dets] == [0]


def test_clear_labels_for_source_unfiltered_clears_every_label_file(tmp_path: Path):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels" / "train").mkdir(parents=True)
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    (source_root / "labels" / "train" / "b.txt").write_text("0 0.5 0.5 0.4 0.2\n")
    (source_root / "classes.txt").write_text("ant\n")

    count = clear_labels_for_source(source_root)

    assert count == 2
    assert (source_root / "labels" / "a.txt").read_text() == ""
    assert (source_root / "labels" / "train" / "b.txt").read_text() == ""
    # Untouched: images dir and classes.txt (at the source root) survive.
    assert (source_root / "classes.txt").read_text() == "ant\n"


def test_clear_labels_for_source_unfiltered_skips_a_stray_classes_txt_under_labels(
    tmp_path: Path,
):
    """Defensive: classes.txt belongs at the source root by convention, but
    if one is ever found under labels/ (e.g. from a manual copy mistake),
    the unfiltered clear must not wipe it -- it isn't a label file."""
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "labels").mkdir(parents=True)
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    (source_root / "labels" / "classes.txt").write_text("ant\n")

    count = clear_labels_for_source(source_root)

    assert count == 1
    assert (source_root / "labels" / "a.txt").read_text() == ""
    assert (source_root / "labels" / "classes.txt").read_text() == "ant\n"


def test_clear_labels_for_source_filtered_clears_only_matching_images(tmp_path: Path):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "a.jpg").write_bytes(b"fake")
    (source_root / "images" / "b.jpg").write_bytes(b"fake")
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    (source_root / "labels" / "b.txt").write_text("0 0.5 0.5 0.4 0.2\n")

    count = clear_labels_for_source(source_root, [source_root / "images" / "a.jpg"])

    assert count == 1
    assert (source_root / "labels" / "a.txt").read_text() == ""
    assert (source_root / "labels" / "b.txt").read_text() != ""  # untouched


def test_clear_labels_for_source_filtered_skips_image_with_no_label_file(
    tmp_path: Path,
):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "unlabeled.jpg").write_bytes(b"fake")

    count = clear_labels_for_source(
        source_root, [source_root / "images" / "unlabeled.jpg"]
    )

    assert count == 0  # no error, just nothing to clear


def test_clear_labels_for_source_unfiltered_on_empty_labels_dir(tmp_path: Path):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "labels").mkdir(parents=True)

    assert clear_labels_for_source(source_root) == 0


def test_clear_labels_for_source_filtered_does_not_cross_split_stem_collision(
    tmp_path: Path,
):
    """Regression for a real bug found in this session's adversarial review:
    find_label_for_image's unanchored recursive-search fallback can resolve
    an unlabeled image in one split to a DIFFERENT split's same-stem label
    file. clear_labels_for_source must never do this -- it must skip an
    image it can't resolve via the mirrored-path or flat-stem strategies,
    not fall back to a wrong file found elsewhere in the tree."""
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images" / "train").mkdir(parents=True)
    (source_root / "images" / "val").mkdir(parents=True)
    (source_root / "labels" / "val").mkdir(parents=True)
    (source_root / "images" / "train" / "f001.jpg").write_bytes(b"fake")  # no label
    (source_root / "images" / "val" / "f001.jpg").write_bytes(b"fake")
    (source_root / "labels" / "val" / "f001.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
    )

    count = clear_labels_for_source(
        source_root, [source_root / "images" / "train" / "f001.jpg"]
    )

    assert count == 0  # nothing resolved -- NOT the val split's label file
    assert (source_root / "labels" / "val" / "f001.txt").read_text() != ""  # untouched


def test_clear_labels_for_source_filtered_dedupes_when_two_images_resolve_same_label(
    tmp_path: Path,
):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "a.jpg").write_bytes(b"fake")
    (source_root / "images" / "a.png").write_bytes(b"fake")  # same stem, different ext
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

    count = clear_labels_for_source(
        source_root,
        [source_root / "images" / "a.jpg", source_root / "images" / "a.png"],
    )

    assert count == 1  # counted once, not twice, for the one file actually cleared
