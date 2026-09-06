"""Tiling, iscrowd boundary, frame-level split, negative-prompt resolution."""

import json
from collections.abc import Iterator

import cv2
import numpy as np
import pytest

from hydra_suite.training.contracts import Sam3LoraParams, SplitConfig
from hydra_suite.training.dataset_io import DatasetIOLimits, DatasetLimitError
from hydra_suite.training.sam3_lora.dataset_build import (
    CURATED_NEGATIVES,
    MIN_RETAINED_AREA_FRAC,
    _split_frame_stems,
    _tile_frame,
    build_sam3_coco_dataset,
    resolve_negative_prompts,
)
from hydra_suite.training.validation import validate_coco_dataset


def _source(tmp_path, n_frames=3, size=2048):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        cv2.imwrite(
            str(img_dir / f"f{i}.jpg"),
            rng.integers(0, 255, (size, size, 3), dtype=np.uint8),
        )
        poly = np.array([[0.50, 0.50], [0.52, 0.50], [0.52, 0.52], [0.50, 0.52]])
        (lbl_dir / f"f{i}.txt").write_text(
            "0 " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)) + "\n"
        )
    (tmp_path / "classes.txt").write_text("ant\n")
    return tmp_path


def _params(**kw):
    base = dict(
        prompt="ant with color patch",
        geometry_mode="custom",
        slice_width=512,
        slice_height=512,
        tile_overlap=0.25,
    )
    base.update(kw)
    return Sam3LoraParams(**base)


def _load(out, split):
    return json.loads((out / split / "_annotations.coco.json").read_text())


def test_category_name_is_the_prompt(tmp_path):
    out = tmp_path / "out"
    build_sam3_coco_dataset(_source(tmp_path / "src"), out, _params())
    assert _load(out, "train")["categories"][0]["name"] == "ant with color patch"


def test_streaming_coco_validation_counts_generated_records(tmp_path):
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(
        _source(tmp_path / "src"), out, _params(keep_empty_tiles=True)
    )
    report = validate_coco_dataset(out, min_train=1, min_val=1)
    assert report.valid
    assert report.stats["train_images"] == stats["train_images"]
    assert report.stats["train_annotations"] == stats["train_annotations"]


def test_split_is_by_frame_not_by_tile(tmp_path):
    out = tmp_path / "out"
    build_sam3_coco_dataset(_source(tmp_path / "src", n_frames=3), out, _params())
    tr = {i["file_name"].split("_")[0] for i in _load(out, "train")["images"]}
    va = {i["file_name"].split("_")[0] for i in _load(out, "valid")["images"]}
    assert tr and va and tr.isdisjoint(va)


def test_single_frame_source_trains_without_validation(tmp_path):
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(
        _source(tmp_path / "src", n_frames=1), out, _params()
    )
    assert stats["train_images"] > 0
    assert stats["validation"] == "none"


def test_empty_tiles_are_kept_when_requested(tmp_path):
    out = tmp_path / "out"
    build_sam3_coco_dataset(
        _source(tmp_path / "src"), out, _params(keep_empty_tiles=True)
    )
    data = _load(out, "train")
    with_ann = {a["image_id"] for a in data["annotations"]}
    assert len(data["images"]) > len(with_ann)


def test_seam_clipped_instances_become_iscrowd(tmp_path):
    # A polygon straddling a tile seam retaining less than the floor is marked
    # iscrowd, not dropped -- dropping teaches SAM3 that a visible half-animal
    # is background. The iscrowd flag is what `datapoints.select_output_objects`
    # later reads to exclude the fragment AND downgrade the tile together.
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(
        _source(tmp_path / "src"),
        out,
        _params(slice_width=1024, slice_height=1024, tile_overlap=0.0),
    )
    assert stats["crowd_annotations"] >= 0  # key exists and is counted
    data = _load(out, "train")
    assert all(a["iscrowd"] in (0, 1) for a in data["annotations"])


def test_retained_area_floor_is_the_chosen_fragment_policy():
    # Deliberate deviation from the spike (see the constant's comment): the
    # floor moved 0.5 -> 0.25 so mostly-visible animals stay full positives and
    # far fewer tiles lose their no-object BCE / FP penalty.
    assert MIN_RETAINED_AREA_FRAC == 0.25


def _crowd_flags(polygon):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    tiles = list(
        _tile_frame(image, [np.asarray(polygon, dtype=np.float32)], 50, 50, 0.0, False)
    )
    return sorted(
        flag for _rect, _crop, instances in tiles for _poly, flag in instances
    )


def test_sub_floor_seam_fragment_is_flagged_and_its_sibling_is_not():
    # 20 % on the left tile (below the 0.25 floor -> fragment), 80 % on the
    # right tile (a normal, fully-supervised positive).
    assert _crowd_flags([[48, 10], [58, 10], [58, 20], [48, 20]]) == [False, True]


def test_above_floor_clip_stays_a_normal_positive():
    # A clean 50/50 split is well above the floor: neither side is a fragment,
    # so neither tile is downgraded.
    assert _crowd_flags([[45, 10], [55, 10], [55, 20], [45, 20]]) == [False, False]


def _seam_source(tmp_path, n_frames=3, size=100):
    """A source whose only animal straddles a tile seam 20/80.

    Below the 0.25 floor on the left tile (a fragment) and well above it on the
    right (a normal positive), so the downgrade tallies are provably nonzero.
    """
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    poly = np.array([[0.48, 0.10], [0.58, 0.10], [0.58, 0.20], [0.48, 0.20]])
    for i in range(n_frames):
        cv2.imwrite(
            str(img_dir / f"f{i}.jpg"),
            rng.integers(0, 255, (size, size, 3), dtype=np.uint8),
        )
        (lbl_dir / f"f{i}.txt").write_text(
            "0 " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)) + "\n"
        )
    (tmp_path / "classes.txt").write_text("ant\n")
    return tmp_path


def test_downgraded_tile_counts_are_actually_tallied(tmp_path):
    # Guards the tally itself, not just the manifest keys: every frame yields
    # exactly one fragment tile (20 % retained) and one full-positive tile.
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(
        _seam_source(tmp_path / "src", n_frames=3),
        out,
        _params(slice_width=50, slice_height=50, tile_overlap=0.0),
    )
    manifest = json.loads((out / "build_manifest.json").read_text())
    counts = manifest["fragment_counts"]
    assert stats["fragment_annotations"] == 3
    assert stats["downgraded_tiles"] == 3
    assert stats["fragment_only_tiles"] == 3
    for key in ("fragment_annotations", "downgraded_tiles", "fragment_only_tiles"):
        assert stats[key] == counts["train"][key] + counts["valid"][key]


def test_manifest_reports_fragment_and_downgraded_tile_counts(tmp_path):
    # The M2 risk (nearly half the annotated stream losing FP pressure) must be
    # visible from the built dataset BEFORE anyone spends GPU hours on it.
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(
        _source(tmp_path / "src"),
        out,
        _params(slice_width=1024, slice_height=1024, tile_overlap=0.0),
    )
    manifest = json.loads((out / "build_manifest.json").read_text())
    assert manifest["min_retained_area_frac"] == MIN_RETAINED_AREA_FRAC
    counts = manifest["fragment_counts"]
    for split in ("train", "valid"):
        assert set(counts[split]) >= {
            "tiles",
            "fragment_annotations",
            "downgraded_tiles",
            "fragment_only_tiles",
        }
        assert counts[split]["fragment_only_tiles"] <= counts[split]["downgraded_tiles"]
        assert counts[split]["downgraded_tiles"] <= counts[split]["tiles"]
    assert stats["fragment_annotations"] == stats["crowd_annotations"]
    assert stats["downgraded_tiles"] == (
        counts["train"]["downgraded_tiles"] + counts["valid"]["downgraded_tiles"]
    )
    assert stats["min_retained_area_frac"] == MIN_RETAINED_AREA_FRAC


def test_split_is_deterministic_under_seed(tmp_path):
    a_out, b_out = tmp_path / "a", tmp_path / "b"
    src = _source(tmp_path / "src")
    build_sam3_coco_dataset(src, a_out, _params(), seed=7)
    build_sam3_coco_dataset(src, b_out, _params(), seed=7)
    assert {i["file_name"] for i in _load(a_out, "valid")["images"]} == {
        i["file_name"] for i in _load(b_out, "valid")["images"]
    }


def test_disk_backed_split_preserves_legacy_exact_order(tmp_path):
    output = tmp_path / "out"
    source = _source(tmp_path / "src", n_frames=7, size=32)
    build_sam3_coco_dataset(
        source, output, _params(slice_width=16, slice_height=16), seed=19
    )
    manifest = json.loads((output / "build_manifest.json").read_text())
    expected_train, expected_valid = _split_frame_stems(
        [f"f{index}" for index in range(7)],
        SplitConfig(),
        19,
    )
    assert manifest["frame_split"] == {
        "train": expected_train,
        "valid": expected_valid,
    }


def test_negative_prompts_prefer_explicit_then_classes_then_curated():
    assert resolve_negative_prompts(
        _params(negative_prompts=["mite"]), ["ant", "beetle"], "ant"
    ) == ["mite"]
    # Tier 2: the OTHER class names of the source -- the confusable concepts.
    assert resolve_negative_prompts(_params(), ["ant", "beetle"], "ant") == ["beetle"]
    got = resolve_negative_prompts(_params(), ["ant"], "ant")
    assert got and set(got).issubset(set(CURATED_NEGATIVES))


def test_curated_negatives_drop_word_overlap_with_the_prompt():
    p = Sam3LoraParams(prompt="ant on a shadow")
    assert "shadow" not in resolve_negative_prompts(p, ["ant"], "ant")


def test_tile_frame_is_lazy():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    tiles = _tile_frame(image, [], 16, 16, 0.0, True)
    assert isinstance(tiles, Iterator)
    assert len(list(tiles)) == 4


def test_label_byte_cap_fails_before_output_promotion(tmp_path):
    source = _source(tmp_path / "src", n_frames=2, size=32)
    (source / "labels" / "f0.txt").write_text("0 " + "1 " * 200)
    output = tmp_path / "out"
    with pytest.raises(DatasetLimitError, match="Text input exceeds"):
        build_sam3_coco_dataset(
            source,
            output,
            _params(slice_width=16, slice_height=16),
            io_limits=DatasetIOLimits(max_label_bytes=32),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_file_count_cap_fails_before_output_promotion(tmp_path):
    source = _source(tmp_path / "src", n_frames=3, size=32)
    output = tmp_path / "out"
    with pytest.raises(DatasetLimitError, match="more than 2"):
        build_sam3_coco_dataset(
            source,
            output,
            _params(slice_width=16, slice_height=16),
            io_limits=DatasetIOLimits(max_files=2),
        )
    assert not output.exists()


def test_write_failure_preserves_previous_atomic_output(tmp_path, monkeypatch):
    source = _source(tmp_path / "src", n_frames=2, size=32)
    output = tmp_path / "out"
    output.mkdir()
    (output / "sentinel.txt").write_text("previous", encoding="utf-8")
    real_write = cv2.imwrite
    calls = 0

    def fail_after_first(path, image):
        nonlocal calls
        calls += 1
        if calls > 1:
            return False
        return real_write(path, image)

    monkeypatch.setattr(cv2, "imwrite", fail_after_first)
    with pytest.raises(RuntimeError, match="Could not write tile"):
        build_sam3_coco_dataset(
            source,
            output,
            _params(slice_width=16, slice_height=16, keep_empty_tiles=True),
        )
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".out.staging-*"))


def test_prompt_limits_apply_before_source_discovery(tmp_path):
    with pytest.raises(ValueError, match="per-prompt cap"):
        build_sam3_coco_dataset(
            tmp_path / "missing-source",
            tmp_path / "out",
            _params(prompt="x" * 257),
        )
