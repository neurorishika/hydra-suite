"""AppleDouble ``._*`` sidecars (written by macOS onto SMB shares) must never be
treated as dataset images or labels."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.data.al.frame_source import ImageFolderFrameSource
from hydra_suite.detectkit.gui.panels.dataset_panel import _copy_tree_without_metadata
from hydra_suite.detectkit.gui.utils import (
    labels_to_clear,
    list_images_in_source,
    source_has_images,
)
from hydra_suite.training.dataset_io import iter_indexed_paths, sorted_file_index
from hydra_suite.utils.hidden_files import is_hidden_file

_APPLEDOUBLE = b"\x00\x05\x16\x07" + b"\x00" * 4092


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    cv2.imwrite(str(images / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    (images / "._a.jpg").write_bytes(_APPLEDOUBLE)
    (images / ".DS_Store").write_bytes(b"\x00")
    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (labels / "._a.txt").write_bytes(_APPLEDOUBLE)
    (labels / "._classes.txt").write_bytes(_APPLEDOUBLE)
    (root / "classes.txt").write_text("ant\n")
    return root


def test_is_hidden_file():
    assert is_hidden_file("x/._a.jpg")
    assert is_hidden_file(".DS_Store")
    assert not is_hidden_file("x/a.jpg")


def test_detectkit_image_listing_skips_appledouble(source: Path):
    assert [p.name for p in list_images_in_source(str(source))] == ["a.jpg"]


def test_source_has_images_ignores_only_appledouble(tmp_path: Path):
    images = tmp_path / "s" / "images"
    images.mkdir(parents=True)
    (images / "._a.jpg").write_bytes(_APPLEDOUBLE)
    assert not source_has_images(str(tmp_path / "s"))


def test_labels_to_clear_skips_appledouble(source: Path):
    assert [p.name for p in labels_to_clear(str(source))] == ["a.txt"]


def test_sorted_file_index_skips_appledouble(source: Path):
    images = source / "images"
    with sorted_file_index(images, suffixes=frozenset({".jpg"})) as index:
        assert [p.name for p in iter_indexed_paths(index, images)] == ["a.jpg"]


def test_staging_copy_drops_appledouble(source: Path, tmp_path: Path):
    dst = tmp_path / "dst"
    _copy_tree_without_metadata(source, dst)
    copied = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*"))
    assert copied == ["classes.txt", "images", "images/a.jpg", "labels", "labels/a.txt"]


def test_al_image_folder_skips_appledouble(source: Path):
    frames = list(ImageFolderFrameSource(str(source / "images")))
    assert len(frames) == 1


def _img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))


def _sidecar(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_APPLEDOUBLE)


@pytest.fixture
def class_folders(tmp_path: Path) -> Path:
    """``train/<cls>/a.jpg`` + an AppleDouble twin, and a hidden class dir."""
    root = tmp_path / "cls"
    for split in ("train", "val"):
        _img(root / split / "ant" / "a.jpg")
        _sidecar(root / split / "ant" / "._a.jpg")
        _sidecar(root / split / ".DS_Store")
    return root


def test_classkit_scan_images_skips_appledouble(source: Path):
    from hydra_suite.classkit.core.data.ingest import scan_images

    assert [p.name for p in scan_images(source)] == ["a.jpg"]
    assert [p.name for p in scan_images(source / "images")] == ["a.jpg"]


def test_classkit_source_import_skips_appledouble(source: Path, class_folders: Path):
    from hydra_suite.classkit.core.data.source_import import (
        _iter_coco_json_candidates,
        _iter_supported_images,
        inspect_external_source,
    )

    assert [p.name for p in _iter_supported_images(source / "images")] == ["a.jpg"]
    _sidecar(source / "._annotations.json")
    (source / "annotations.json").write_text("{}")
    assert [p.name for p in _iter_coco_json_candidates(source)] == ["annotations.json"]
    (class_folders / "train" / ".AppleDouble").mkdir()
    _img(class_folders / "train" / ".AppleDouble" / "x.jpg")
    inspection = inspect_external_source(class_folders)
    assert inspection.source_kind == "class_folders"
    assert inspection.images_count == 2
    assert inspection.discovered_labels == ["ant"]


def test_classkit_source_validation_skips_appledouble(source: Path):
    from hydra_suite.classkit.gui.dialogs.source_validation import list_classkit_images

    assert [p.name for p in list_classkit_images(source / "images")] == ["a.jpg"]


def test_detectkit_coco_candidates_skip_appledouble(source: Path):
    from hydra_suite.detectkit.gui.source_import import _iter_coco_json_candidates

    _sidecar(source / "._x.coco.json")
    _sidecar(source / "annotations" / "._instances.json")
    assert _iter_coco_json_candidates(source) == []


def test_posekit_image_listing_skips_appledouble(source: Path):
    from hydra_suite.posekit.gui.dialogs.utils import list_images_in_dir
    from hydra_suite.posekit.gui.utils import list_images

    assert [p.name for p in list_images(source / "images")] == ["a.jpg"]
    assert [p.name for p in list_images_in_dir(source / "images")] == ["a.jpg"]


def test_posekit_label_migration_skips_appledouble(source: Path):
    from hydra_suite.posekit.core.extensions import migrate_labels_keypoints

    (source / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.5 0.5 2\n")
    _, total = migrate_labels_keypoints(source / "labels", ["h"], ["h", "t"])
    assert total == 1
    assert (source / "labels" / "._a.txt").read_bytes() == _APPLEDOUBLE


def test_filterkit_collect_images_skips_appledouble(source: Path, class_folders):
    from hydra_suite.filterkit.core import FilterKitCore, collect_images_for_root

    kind, paths = collect_images_for_root(source / "images")
    assert (kind, [p.name for p in paths]) == ("images", ["a.jpg"])
    kind, paths = collect_images_for_root(class_folders)
    assert kind == "class_folders"
    assert sorted(p.name for p in paths) == ["a.jpg", "a.jpg"]
    assert FilterKitCore.is_supported_video("v/clip.mp4")
    assert not FilterKitCore.is_supported_video("v/._clip.mp4")


def test_training_classify_samples_skip_appledouble(class_folders: Path):
    from hydra_suite.training.canonical_transform import is_visible_image_file
    from hydra_suite.training.runner import _iter_classify_samples

    samples = list(_iter_classify_samples(class_folders, "train"))
    assert [p.name for p, _ in samples] == ["a.jpg"]
    assert is_visible_image_file("x/a.JPG")
    assert not is_visible_image_file("x/._a.jpg")
    assert not is_visible_image_file("x/a.txt")

    from torchvision import datasets

    ds = datasets.ImageFolder(
        str(class_folders / "train"), is_valid_file=is_visible_image_file
    )
    assert [Path(p).name for p, _ in ds.samples] == ["a.jpg"]


def test_multihead_dataset_skips_appledouble(tmp_path: Path):
    from hydra_suite.training.multihead_dataset import MultiFactorImageFolder

    _img(tmp_path / "a__x" / "f.jpg")
    _sidecar(tmp_path / "a__x" / "._f.jpg")
    (tmp_path / ".AppleDouble").mkdir()
    ds = MultiFactorImageFolder(
        str(tmp_path), class_names_per_factor=[["a"], ["x"]], delimiter="__"
    )
    assert len(ds) == 1


def test_dataset_merge_skips_appledouble(source: Path):
    from hydra_suite.data.dataset_merge import _collect_images, validate_labels

    assert [Path(p).name for p in _collect_images(source / "images")] == ["a.jpg"]
    obb = source / "obb_labels"
    obb.mkdir()
    (obb / "a.txt").write_text("0 0 0 1 0 1 1 0 1\n")
    _sidecar(obb / "._a.txt")
    assert validate_labels(obb) == ({0}, 1)


def test_dataset_inspector_list_split_skips_appledouble(source: Path):
    from hydra_suite.training.dataset_inspector import _collect_list_split

    listing = source / "train.txt"
    listing.write_text("images/a.jpg\nimages/._a.jpg\n")
    items = _collect_list_split(source, listing, "train")
    assert [Path(item.image_path).name for item in items] == ["a.jpg"]


def test_trackerkit_validate_dataset_skips_appledouble(source: Path):
    from hydra_suite.trackerkit.gui.validate_labels import validate_dataset

    valid, invalid, _ = validate_dataset(source / "labels", kpt_count=0)
    assert valid + invalid == 1


def test_trackerkit_model_test_samples_skip_appledouble(source: Path):
    from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import (
        _collect_sample_images,
    )

    assert [Path(p).name for p in _collect_sample_images(str(source))] == ["a.jpg"]


def _fingerprints_ignore_sidecar(fingerprint, root: Path, sidecar: Path) -> None:
    before = fingerprint(root)
    _sidecar(sidecar)
    (sidecar.parent / ".DS_Store").write_bytes(b"\x00")
    assert fingerprint(root) == before


def test_dataset_fingerprint_ignores_appledouble(source: Path):
    from hydra_suite.training.registry import dataset_fingerprint

    _fingerprints_ignore_sidecar(
        dataset_fingerprint, source, source / "images" / "._b.jpg"
    )


def test_directory_content_id_ignores_appledouble(source: Path):
    from hydra_suite.core.inference.content_id import directory_content_id

    _fingerprints_ignore_sidecar(
        lambda root: directory_content_id(str(root)),
        source,
        source / "images" / "._b.jpg",
    )


def test_detectkit_source_footprint_ignores_appledouble(source: Path):
    from hydra_suite.detectkit.jobs.dataset_preparation_sidecar import (
        _scan_source_footprint,
    )
    from hydra_suite.training.contracts import SourceDataset

    sources = (SourceDataset(path=str(source)),)
    _fingerprints_ignore_sidecar(
        lambda _root: _scan_source_footprint(sources),
        source,
        source / "images" / "._b.jpg",
    )


def test_job_model_copy_drops_appledouble(tmp_path: Path):
    from hydra_suite.data.tracking_job.references import (
        PlannedModel,
        copy_model_reference,
    )

    model = tmp_path / "sleap_run"
    model.mkdir()
    (model / "best.ckpt").write_bytes(b"w")
    _sidecar(model / "._best.ckpt")
    job = copy_model_reference(
        PlannedModel(role="pose", source_path=str(model), kind="directory", key="m"),
        tmp_path / "models",
    )
    assert job.files == ["m/best.ckpt"]


def test_sleap_export_dir_resolution_skips_appledouble(tmp_path: Path):
    from hydra_suite.core.individual.pose.backends.sleap import (
        _resolve_export_model_path,
    )

    (tmp_path / "model.onnx").write_bytes(b"onnx")
    _sidecar(tmp_path / "._model.onnx")
    assert _resolve_export_model_path(str(tmp_path), "onnx").name == "model.onnx"
