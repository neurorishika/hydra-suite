"""Before-gate for the multi-scale SAM3 port (plan Task 1).

Freezes the CURRENT single-scale ``build_sam3_coco_dataset`` output as a
committed tree hash. Task 3 adds a scale-set path whose documented contract is
that ``object_tile_fractions=()`` / ``full_frame_mix=False`` fork to *this*
exact code path; this golden is the only thing that can prove that claim
instead of asserting it.

The golden is a characterization artifact: it has no "fails first" phase, and
it must be re-run UNMODIFIED after every later task. Regenerate deliberately
only when a behaviour change is intended and reviewed:

    HYDRA_UPDATE_SAM3_GOLDEN=1 python -m pytest tests/test_sam3_multiscale_gate.py

Encoder note: tile images are JPEG-encoded by OpenCV, so the byte hashes are
pinned to the libjpeg build recorded in ``ATTRIBUTION.md``. A mismatch of the
image hashes ALONE, with identical paths and identical COCO/manifest hashes,
is an encoder difference rather than a builder behaviour change.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.contracts import Sam3LoraParams, SplitConfig
from hydra_suite.training.sam3_lora.dataset_build import build_sam3_coco_dataset

GOLDEN_DIR = Path(__file__).parent / "data" / "sam3_multiscale_golden"
GOLDEN_PATH = GOLDEN_DIR / "single_scale_tree.json"

# Deterministic tiny corpus. Frames are written as PNG (lossless) so the only
# lossy encode in the whole pipeline is the builder's own tile JPEG.
CORPUS_SEED = 20260906
FRAME_COUNT = 4
FRAME_SIZE = 2048
BODY_PX = 40  # -> reference_body_px ~ 40, tile ~ 40/0.055 ~ 727 px

VOLATILE_MANIFEST_KEYS = ("created_at", "source")


def write_corpus(root: Path) -> Path:
    """Emit the committed-by-recipe synthetic source under ``root``."""
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rng = np.random.default_rng(CORPUS_SEED)
    step = BODY_PX / FRAME_SIZE
    for index in range(FRAME_COUNT):
        frame = rng.integers(0, 255, (FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        assert cv2.imwrite(str(images / f"frame{index:02d}.png"), frame)
        lines = []
        # Three bodies per frame at fixed, frame-varying positions: enough to
        # populate several tiles and to cross at least one tile seam.
        for slot in range(3):
            cx = 0.2 + 0.3 * slot + 0.05 * index
            cy = 0.25 + 0.2 * slot
            poly = np.array(
                [
                    [cx - step / 2, cy - step / 2],
                    [cx + step / 2, cy - step / 2],
                    [cx + step / 2, cy + step / 2],
                    [cx - step / 2, cy + step / 2],
                ]
            )
            lines.append("0 " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)))
        (labels / f"frame{index:02d}.txt").write_text("\n".join(lines) + "\n")
    (root / "classes.txt").write_text("ant\n")
    return root


def default_params() -> Sam3LoraParams:
    """Contract defaults -- the single-scale geometry the golden freezes."""
    return Sam3LoraParams(prompt="ant")


def build_single_scale(tmp_path: Path, name: str) -> Path:
    out = tmp_path / name
    build_sam3_coco_dataset(
        str(write_corpus(tmp_path / f"src_{name}")),
        str(out),
        default_params(),
        seed=42,
        split=SplitConfig(),
    )
    return out


def _normalise_json(path: Path) -> bytes:
    payload = json.loads(path.read_text())
    if path.name == "build_manifest.json":
        for key in VOLATILE_MANIFEST_KEYS:
            payload.pop(key, None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def tree_hash(root: Path) -> dict[str, str]:
    """Sorted relative path -> SHA-256, with volatile manifest keys removed."""
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".json":
            blob = _normalise_json(path)
        else:
            blob = path.read_bytes()
        digests[rel] = hashlib.sha256(blob).hexdigest()
    return digests


def test_single_scale_build_is_deterministic(tmp_path):
    """Two builds of the same corpus agree byte-for-byte (golden precondition)."""
    first = tree_hash(build_single_scale(tmp_path, "a"))
    second = tree_hash(build_single_scale(tmp_path, "b"))
    assert first == second
    assert first, "golden corpus produced no files"


def test_single_scale_golden(tmp_path):
    """The current single-scale dataset tree matches the committed golden."""
    actual = tree_hash(build_single_scale(tmp_path, "gold"))
    if os.environ.get("HYDRA_UPDATE_SAM3_GOLDEN"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    expected = json.loads(GOLDEN_PATH.read_text())
    assert sorted(actual) == sorted(expected), "dataset tree LAYOUT changed"
    assert actual == expected, "dataset tree CONTENT changed"
