"""Task 3: the SAM3 builder resolves a scale SET (defaults unchanged).

The default-path guarantee is proven by ``test_sam3_multiscale_gate.py``'s
committed tree hash, not here. These tests cover the opt-in path: the fan-out,
its naming, its ``scale_group`` DATA field, the dedup, the full-frame arm, and
the loud refusal on a per-scale tile-count blow-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydra_suite.training.contracts import Sam3LoraParams, SplitConfig
from hydra_suite.training.sam3_lora.dataset_build import build_sam3_coco_dataset
from hydra_suite.training.ultralytics_scale_balance import scale_group_for_path
from hydra_suite.utils.slice_geometry import plan_tiles
from tests.test_sam3_multiscale_gate import write_corpus


def _build(tmp_path: Path, name: str, params: Sam3LoraParams) -> Path:
    out = tmp_path / name
    build_sam3_coco_dataset(
        str(write_corpus(tmp_path / f"src_{name}")),
        str(out),
        params,
        seed=42,
        split=SplitConfig(),
    )
    return out


def _coco(out: Path, split: str = "train") -> dict:
    return json.loads((out / split / "_annotations.coco.json").read_text())


def _params(**kwargs) -> Sam3LoraParams:
    return Sam3LoraParams(prompt="ant", **kwargs)


def test_two_fractions_emit_both_scales(tmp_path):
    """Each fraction contributes its own tiles, named and tagged per scale."""
    out = _build(tmp_path, "two", _params(object_tile_fractions=(0.055, 0.02)))
    records = _coco(out)["images"]
    groups = {record["scale_group"] for record in records}
    assert len(groups) == 2, groups
    for record in records:
        # scale_group is DATA, but it must still agree with the filename token
        # so the Ultralytics-side parser and a field read never disagree (D19).
        assert record["scale_group"] == scale_group_for_path(record["file_name"])
        assert isinstance(record["tile_px"], list) and len(record["tile_px"]) == 2
    assert len({record["file_name"] for record in records}) == len(records)


def test_per_scale_tile_counts_match_plan_tiles(tmp_path):
    """The fan-out plans exactly ``plan_tiles`` tiles at each resolved scale."""
    params = _params(object_tile_fractions=(0.055, 0.02))
    out = _build(tmp_path, "counts", params)
    manifest = json.loads((out / "build_manifest.json").read_text())
    stems = manifest["frame_split"]["train"]
    expected: dict[str, int] = {}
    for width, height in (
        # frames are square FRAME_SIZE; sizes come from the manifest's own set
        (size[0], size[1])
        for size in _resolved_scales(out)
    ):
        plan = plan_tiles((2048, 2048), width, height, 0.25, 0.25)
        expected[f"tile:{width}x{height}"] = len(plan.tiles) * len(stems)
    actual: dict[str, int] = {}
    for record in _coco(out)["images"]:
        actual[record["scale_group"]] = actual.get(record["scale_group"], 0) + 1
    assert actual == expected


def _resolved_scales(out: Path) -> list[tuple[int, int]]:
    seen: list[tuple[int, int]] = []
    for record in _coco(out)["images"]:
        group = record["scale_group"]
        if not group.startswith("tile:"):
            continue
        width, height = group[len("tile:") :].split("x")
        size = (int(width), int(height))
        if size not in seen:
            seen.append(size)
    return seen


def test_duplicate_fractions_do_not_duplicate_tiles(tmp_path):
    """Fractions that round to one tile size produce ONE copy of those tiles."""
    single = _build(tmp_path, "single", _params(object_tile_fractions=(0.055,)))
    duplicated = _build(
        tmp_path, "dup", _params(object_tile_fractions=(0.055, 0.055, 0.0550001))
    )
    assert len(_coco(duplicated)["images"]) == len(_coco(single)["images"])


def test_full_frame_mix_adds_one_full_arm_per_frame(tmp_path):
    """``full_frame_mix`` adds exactly one un-tiled copy per frame."""
    out = _build(tmp_path, "full", _params(full_frame_mix=True))
    records = _coco(out)["images"]
    manifest = json.loads((out / "build_manifest.json").read_text())
    fulls = [r for r in records if r["scale_group"] == "full"]
    assert len(fulls) == len(manifest["frame_split"]["train"])
    assert all(r["file_name"].endswith("_full.jpg") for r in fulls)
    assert all(r["width"] == 2048 and r["height"] == 2048 for r in fulls)
    # full_frame_mix alone still forks off the legacy naming.
    assert not any("_tile" in r["file_name"] for r in records)


def test_tile_ceiling_refuses_loudly_per_scale(tmp_path):
    """MAX_TILES_PER_FRAME is per scale and SAM3 must RAISE, never truncate.

    The YOLO builder swallows this ``ValueError`` and drops the offending
    scale; doing that here would silently build a dataset with fewer scales
    than the run claims, corrupting the very comparison this port exists for.
    """
    params = _params(object_tile_fractions=(0.055, 0.9), tile_overlap=0.99)
    with pytest.raises(ValueError, match="tile"):
        _build(tmp_path, "ceiling", params)


def test_stem_collision_guard_still_fires(tmp_path):
    """The duplicate-stem guard is unaffected by the scale-set path."""
    root = write_corpus(tmp_path / "src_collide")
    duplicate = root / "images" / "frame00.jpeg"
    duplicate.write_bytes((root / "images" / "frame00.png").read_bytes())
    (root / "labels" / "frame00.txt").read_text()
    with pytest.raises(RuntimeError, match="duplicate image stems"):
        build_sam3_coco_dataset(
            str(root),
            str(tmp_path / "collide_out"),
            _params(object_tile_fractions=(0.055, 0.02)),
            seed=42,
            split=SplitConfig(),
        )


def _plan_dict(**sam3) -> dict:
    return {
        "version": 1,
        "workspace": "/tmp/ws",
        "sources": [{"path": "."}],
        "class_names": ["ant"],
        "roles": [{"role": "semantic_sam3"}],
        "sam3": {
            "prompt": "ant",
            "label_quality_acknowledged": True,
            **sam3,
        },
    }


def test_plan_round_trip_keeps_the_scale_set():
    """The set must survive the JSON plan boundary, not just a direct call."""
    from hydra_suite.detectkit.config.training import DetectTrainingPlan

    plan = DetectTrainingPlan.from_dict(
        _plan_dict(object_tile_fractions=[0.05, 0.1], full_frame_mix=True)
    )
    assert plan.sam3_params.object_tile_fractions == (0.05, 0.1)
    assert plan.sam3_params.full_frame_mix is True
    reloaded = DetectTrainingPlan.from_dict(plan.to_dict())
    assert reloaded.sam3_params.object_tile_fractions == (0.05, 0.1)
    assert reloaded.sam3_params.full_frame_mix is True


@pytest.mark.parametrize("bad", ([0.0], [-0.1], [1.5]))
def test_plan_rejects_out_of_band_fractions(bad):
    from hydra_suite.detectkit.config.training import (
        DetectTrainingPlan,
        TrainingPlanError,
    )

    with pytest.raises(TrainingPlanError, match="object_tile_fractions"):
        DetectTrainingPlan.from_dict(_plan_dict(object_tile_fractions=bad)).validate()


# ---------------------------------------------------------------------------
# Task 7 -- per-scale counters, including `non_square_tiles` (R2 + R3b).
#
# A partial edge tile is not square, and `datapoints.py:167-168` stretches it
# anisotropically to RESxRES. Adding scales multiplies the edge-tile
# population, so a multi-scale dataset carries proportionally MORE distorted
# supervision than a single-scale one -- and nothing reported it.
# ---------------------------------------------------------------------------


def _rect_corpus(root: Path, width: int, height: int) -> Path:
    """A corpus whose frames are NOT square, so edge tiles are not square."""
    import cv2
    import numpy as np

    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rng = np.random.default_rng(31415)
    for index in range(2):
        frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        assert cv2.imwrite(str(images / f"frame{index:02d}.png"), frame)
        step_x, step_y = 40.0 / width, 40.0 / height
        lines = []
        for slot in range(3):
            cx = 0.2 + 0.3 * slot
            cy = 0.25 + 0.2 * slot + 0.02 * index
            poly = [
                [cx - step_x / 2, cy - step_y / 2],
                [cx + step_x / 2, cy - step_y / 2],
                [cx + step_x / 2, cy + step_y / 2],
                [cx - step_x / 2, cy + step_y / 2],
            ]
            lines.append("0 " + " ".join(f"{v:.6f}" for point in poly for v in point))
        (labels / f"frame{index:02d}.txt").write_text("\n".join(lines) + "\n")
    (root / "classes.txt").write_text("ant\n")
    return root


def _build_rect(
    tmp_path: Path,
    name: str,
    params: Sam3LoraParams,
    *,
    size: tuple[int, int] = (1500, 1100),
) -> tuple[Path, dict]:
    out = tmp_path / name
    summary = build_sam3_coco_dataset(
        str(_rect_corpus(tmp_path / f"rect_{name}", *size)),
        str(out),
        params,
        seed=42,
        split=SplitConfig(),
    )
    return out, summary


def test_non_square_tiles_are_counted_per_scale(tmp_path):
    """The counter equals what `plan_tiles` says is clipped, at each scale."""
    params = _params(object_tile_fractions=(0.055, 0.02))
    out, _summary = _build_rect(tmp_path, "nonsq", params)
    manifest = json.loads((out / "build_manifest.json").read_text())
    counts = manifest["scale_counts"]["train"]
    stems = len(manifest["frame_split"]["train"])
    assert stems

    expected = {}
    for width, height in (tuple(pair) for pair in manifest["tile_px_set"]):
        plan = plan_tiles((1100, 1500), width, height, 0.25, 0.25)
        clipped = sum(
            1
            for tile in plan.tiles
            for w, h in [(tile[2] - tile[0], tile[3] - tile[1])]
            if w != h
        )
        expected[f"tile:{width}x{height}"] = clipped * stems

    actual = {name: bucket["non_square_tiles"] for name, bucket in counts.items()}
    assert actual == expected
    assert any(value > 0 for value in expected.values()), "corpus grew no edge tiles"


def test_full_frames_are_counted_as_non_square_when_they_are(tmp_path):
    out, _summary = _build_rect(
        tmp_path,
        "fullnonsq",
        _params(object_tile_fractions=(0.055,), full_frame_mix=True),
    )
    manifest = json.loads((out / "build_manifest.json").read_text())
    full = manifest["scale_counts"]["train"]["full"]
    assert full["non_square_tiles"] == full["tiles"] > 0


def test_non_square_total_is_reported_on_the_single_scale_path_too(tmp_path):
    """No scale set, so no per-scale table -- but the total still travels.

    The build manifest is byte-frozen by the Task 1 golden, so this rides on
    the builder's RETURN summary rather than a new manifest key. The frame is
    SHORTER than the resolved tile (727px), which is the shape that actually
    produces a clipped, anisotropically-stretched tile: `plan_tiles` shifts an
    interior last tile back to full size, so a merely-not-a-multiple frame does
    NOT (that is a fact about the planner worth pinning here).
    """
    out, summary = _build_rect(tmp_path, "single", _params(), size=(1500, 600))
    assert summary["non_square_tiles"] > 0
    (square,) = {
        (record["width"], record["height"])
        for record in _coco(out)["images"]
        if record["width"] == record["height"]
    } or {None}
    assert square is None, "expected every tile clipped by the short frame"


def test_per_scale_table_is_logged_before_any_gpu_time(tmp_path, caplog):
    with caplog.at_level("INFO"):
        _build_rect(tmp_path, "logged", _params(object_tile_fractions=(0.055, 0.02)))
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "non_square_tiles" in text
    assert "tile:" in text
