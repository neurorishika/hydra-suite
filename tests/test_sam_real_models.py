"""Real-model SAM2 / SAM3 inference smoke on every device this host has.

Marked ``slow`` (excluded by default). Run with::

    pytest -m slow tests/test_sam_real_models.py

Skips -- never downloads -- when weights or the fixture clip are missing, so
it is safe anywhere; on a machine with the SAM3 checkpoint it proves the
inference path runs on CPU, MPS and CUDA alike.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

# Fixture clips are not committed; HYDRA_EQUIV_FIXTURES points a worktree at a
# checkout that has them.
CLIP = (
    Path(
        os.environ.get(
            "HYDRA_EQUIV_FIXTURES",
            Path(__file__).resolve().parents[1] / "tools/equivalence/fixtures",
        )
    )
    / "clips/fly_obb.mp4"
)
# A fly in frame 0 of the fixture clip (from the equivalence tracking output).
FLY_XY = (499, 303)


def _devices() -> list[str]:
    from hydra_suite.core.inference.torch_device import device_choices

    return [value for _label, value in device_choices() if value != "auto"]


@pytest.fixture(scope="module")
def fly_frame() -> np.ndarray:
    import cv2

    if not CLIP.exists():
        pytest.skip(f"fixture clip missing: {CLIP}")
    cap = cv2.VideoCapture(str(CLIP))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        pytest.skip("could not decode the fixture clip")
    return frame


@pytest.mark.parametrize("device", _devices())
def test_sam2_segments_a_real_fly(fly_frame, device):
    pytest.importorskip("sam2")
    from hydra_suite.core.inference.sam2.executor import Sam2SegmentExecutor

    try:
        executor = Sam2SegmentExecutor.from_variant(
            "sam2.1-hiera-tiny", device=device, allow_download=False
        )
    except ValueError as exc:  # not downloaded and downloads disabled
        pytest.skip(str(exc))
    x, y = FLY_XY
    executor.set_image(fly_frame[:, :, ::-1].copy())
    mask, iou = executor.segment((x - 60, y - 60, x + 60, y + 60), [(x, y)], [])
    assert mask.shape == fly_frame.shape[:2]
    assert mask[y, x], "the prompted fly centre must be inside its own mask"
    assert 50 < int(mask.sum()) < 120 * 120
    assert iou > 0.5


@pytest.mark.parametrize("device", _devices())
def test_sam3_finds_flies_by_text(fly_frame, device):
    from hydra_suite.core.inference.semantic import checkpoints

    if not checkpoints.probe_dependencies().usable:
        pytest.skip(checkpoints.probe_dependencies().reason)
    if not checkpoints.checkpoint_path("sam3").exists():
        pytest.skip("SAM3 checkpoint not downloaded (licence-gated)")
    from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

    labeler = Sam3SemanticLabeler.from_variant(
        "sam3", device=device, allow_download=False
    )
    instances = labeler.label_image(fly_frame, "fly", confidence_threshold=0.35)
    assert len(instances) >= 1
