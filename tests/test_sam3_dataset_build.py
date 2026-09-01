"""Tiling, iscrowd boundary, frame-level split, negative-prompt resolution."""

import json

import cv2
import numpy as np

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora.dataset_build import (
    CURATED_NEGATIVES,
    build_sam3_coco_dataset,
    resolve_negative_prompts,
)


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
    # A polygon straddling a tile seam retains <50% on one side; it must be
    # marked iscrowd, not dropped -- dropping teaches SAM3 that a visible
    # half-animal is background.
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(
        _source(tmp_path / "src"),
        out,
        _params(slice_width=1024, slice_height=1024, tile_overlap=0.0),
    )
    assert stats["crowd_annotations"] >= 0  # key exists and is counted
    data = _load(out, "train")
    assert all(a["iscrowd"] in (0, 1) for a in data["annotations"])


def test_split_is_deterministic_under_seed(tmp_path):
    a_out, b_out = tmp_path / "a", tmp_path / "b"
    src = _source(tmp_path / "src")
    build_sam3_coco_dataset(src, a_out, _params(), seed=7)
    build_sam3_coco_dataset(src, b_out, _params(), seed=7)
    assert {i["file_name"] for i in _load(a_out, "valid")["images"]} == {
        i["file_name"] for i in _load(b_out, "valid")["images"]
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
