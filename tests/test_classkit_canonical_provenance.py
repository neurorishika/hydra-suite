import json

from hydra_suite.classkit.core.data.canonical_provenance import (
    canonical_geometry_for_training_images,
)
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry


def _write_provenance(dataset_dir, geometry: CanonicalGeometry) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "metadata.json").write_text(
        json.dumps({"parameters": {"canonical": geometry.to_dict()}}),
        encoding="utf-8",
    )


def test_single_source_root_recovers_geometry(tmp_path):
    src = tmp_path / "export1"
    geom = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    _write_provenance(src, geom)

    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    metadata_by_path = {
        str(img_a): {"source_root": str(src)},
        str(img_b): {"source_root": str(src)},
    }
    result = canonical_geometry_for_training_images([img_a, img_b], metadata_by_path)
    assert result == geom


def test_disagreeing_source_roots_yield_none(tmp_path):
    src1 = tmp_path / "export1"
    src2 = tmp_path / "export2"
    _write_provenance(src1, CanonicalGeometry.from_reference(20.0, 2.44, 1.5))
    _write_provenance(src2, CanonicalGeometry.from_reference(20.0, 2.0, 1.3))

    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    metadata_by_path = {
        str(img_a): {"source_root": str(src1)},
        str(img_b): {"source_root": str(src2)},
    }
    assert (
        canonical_geometry_for_training_images([img_a, img_b], metadata_by_path) is None
    )


def test_missing_source_root_yields_none(tmp_path):
    img_a = tmp_path / "a.png"
    assert canonical_geometry_for_training_images([img_a], {str(img_a): {}}) is None


def test_unstamped_source_dataset_yields_none(tmp_path):
    src = tmp_path / "export1"
    src.mkdir()
    img_a = tmp_path / "a.png"
    metadata_by_path = {str(img_a): {"source_root": str(src)}}
    assert canonical_geometry_for_training_images([img_a], metadata_by_path) is None


def test_empty_image_list_yields_none(tmp_path):
    assert canonical_geometry_for_training_images([], {}) is None
